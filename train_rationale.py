#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""SFT + GRPO on structured clinical rationales (RLVR regime), separate from TimeCap.

Two stages on OpenTSLMSP, reusing CurriculumTrainer's model, checkpoint and LoRA plumbing
without changing ``curriculum_learning.py``:

  rationale_sft   teacher-forced cross-entropy on rationales (export_rationale_sft.py)
                  so the model learns the <findings>/<reasoning>/<answer> format
  rationale_grpo  GRPO on label-only prompts (export_rationale_grpo.py), warm-started
                  from the SFT checkpoint; reward = MHCRationaleQADataset.reward

GRPO is ported from OpenTSLM-R1 (branch reinforce-sp): K samples per prompt, advantage =
reward - group mean (no std normalisation), token-level loss over the group, no KL, no
clipping, groups with identical rewards skipped. Differences from R1:
  - the prompt is re-embedded with gradients in the loss, so encoder and projector train
    (R1 reused embeddings from a no_grad pass, so only LoRA received gradients)
  - the loss covers generated tokens up to and including the first EOS (R1 masked pad ==
    EOS, so stopping was never reinforced)
  - sampling uses the plain softmax at the given temperature (no top-k / top-p from the
    LLM's generation config) and the loss uses the same tempered distribution
  - K is processed in chunks to bound activation memory

Evaluation generates greedily and reports accuracy, balanced accuracy, format rate and
AUROC per label; the AUROC score is log p("yes") - log p("no") at the model's own
<answer> tag.

Example::

    python train_rationale.py --stages rationale_sft rationale_grpo \\
        --sft_file rationale_sft_v4.arrow --grpo_file rationale_grpo_v1.arrow
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from curriculum_learning import CurriculumTrainer, StageConfig
from opentslm.model_config import GRAD_CLIP_NORM, PATCH_SIZE
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from time_series_datasets.mhc_rationale_qa_dataset import MHCRationaleQADataset


@dataclass
class GRPOConfig(StageConfig):
    num_samples: int = 8
    temperature: float = 1.0
    grad_accumulation_steps: int = 8
    pg_chunk_size: int = 2
    eval_every: int = 0  # optimizer steps between validations; 0 = once per epoch


# Model helpers (OpenTSLMSP)

def _prep(sample: dict) -> list[dict]:
    batch = [sample]
    extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=PATCH_SIZE)
    return batch


def completion_mask(ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """1 for tokens up to and including the first EOS of each row, 0 after."""
    is_eos = ids == eos_id
    first = torch.where(is_eos.any(1), is_eos.int().argmax(1), torch.full_like(ids[:, 0], ids.size(1)))
    return (torch.arange(ids.size(1), device=ids.device)[None] <= first[:, None]).long()


def token_logprobs(model, sample: dict, ids: torch.Tensor, mask: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
    """Per-token log-probs [K, T] of continuations ``ids`` after the sample's prompt."""
    prompt, prompt_mask = model.pad_and_apply_batch([sample])
    k, t = ids.shape
    embeds = torch.cat([prompt.expand(k, -1, -1), model.llm.get_input_embeddings()(ids)], dim=1)
    attn = torch.cat([prompt_mask.expand(k, -1), mask], dim=1)
    logits = model.llm(inputs_embeds=embeds, attention_mask=attn, logits_to_keep=t + 1,
                       return_dict=True).logits[:, :-1].float()
    logp = torch.log_softmax(logits / temperature, dim=-1).gather(2, ids.unsqueeze(-1)).squeeze(-1)
    return logp * mask


@torch.no_grad()
def generate(model, sample: dict, max_new_tokens: int, num_samples: int = 1,
             temperature: Optional[float] = None) -> torch.Tensor:
    """New token ids [K, T]; greedy when ``temperature`` is None."""
    prompt, prompt_mask = model.pad_and_apply_batch([sample])
    tok = model.tokenizer
    sampling = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if temperature else dict(do_sample=False)
    return model.llm.generate(
        inputs_embeds=prompt.expand(num_samples, -1, -1).contiguous(),
        attention_mask=prompt_mask.expand(num_samples, -1).contiguous(),
        max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        **sampling,
    )


@torch.no_grad()
def answer_score(model, sample: dict, generated: str) -> float:
    """log p(yes) - log p(no) for the answer following the model's own rationale."""
    cut = generated.rfind("<answer>")
    prefix = generated[:cut] if cut >= 0 else generated.rstrip() + "\n"
    # Answers are tokenized with special tokens in SFT (compute_loss), so score them the same way.
    seqs = [torch.tensor(model.tokenizer(prefix + f"<answer>{a}</answer>").input_ids) for a in ("yes", "no")]
    ids = pad_sequence(seqs, batch_first=True, padding_value=model.tokenizer.pad_token_id).to(model.device)
    mask = pad_sequence([torch.ones_like(s) for s in seqs], batch_first=True).to(model.device)
    lp = token_logprobs(model, sample, ids, mask).sum(1)
    return float(lp[0] - lp[1])


def answer_metrics(predictions: list[dict]) -> dict:
    """Accuracy, balanced accuracy, format rate and AUROC overall and per label."""
    def block(preds):
        gold = np.array([p["gold"] == "yes" for p in preds])
        pred = np.array([p["pred"] == "yes" for p in preds])
        valid = np.array([p["pred"] is not None for p in preds])
        # Unparseable answers count as wrong for both classes.
        tpr = float(np.sum(pred & gold & valid) / gold.sum()) if gold.any() else None
        tnr = float(np.sum(~pred & ~gold & valid) / (~gold).sum()) if (~gold).any() else None
        rates = [r for r in (tpr, tnr) if r is not None]
        out = {"n": len(preds), "n_pos": int(gold.sum()),
               "accuracy": float(np.mean((pred == gold) & valid)),
               "balanced_accuracy": float(np.mean(rates)),
               "answer_rate": float(valid.mean()),
               "format_rate": float(np.mean([p["well_formed"] for p in preds])),
               "mean_reward": float(np.mean([p["reward"] for p in preds]))}
        scores = [p.get("score") for p in preds]
        if 0 < gold.sum() < len(gold) and all(s is not None for s in scores):
            out["auroc"] = float(roc_auc_score(gold, scores))
        return out

    by_label = defaultdict(list)
    for p in predictions:
        by_label[p["label"]].append(p)
    per_label = {label: block(preds) for label, preds in sorted(by_label.items())}
    metrics = {"overall": block(predictions), "per_label": per_label}
    for key in ("balanced_accuracy", "auroc"):
        values = [m[key] for m in per_label.values() if key in m]
        if values:
            metrics[f"macro_{key}"] = float(np.mean(values))
    return metrics


class RationaleTrainer(CurriculumTrainer):
    def __init__(self, *args, max_new_tokens: int = 512, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    # Evaluation (used by both stages)

    def _predict(self, dataset, desc: str, score: bool = True) -> list[dict]:
        self.model.eval()
        predictions = []
        for sample in tqdm(dataset, desc=desc, dynamic_ncols=True):
            batch = _prep(sample)
            ids = generate(self.model, sample, self.max_new_tokens)
            text = self.model.tokenizer.decode(ids[0], skip_special_tokens=True)
            pred = dataset.extract_answer(text)
            predictions.append({
                "sample_id": sample["sample_id"], "label": sample["label"], "gold": sample["gold"],
                "pred": pred, "well_formed": dataset.is_well_formed(text),
                "reward": dataset.reward(sample, text),
                "score": answer_score(self.model, sample, text) if score else None,
                "generated": text,
                "loss": float(self.model.compute_loss(batch)) if sample["has_rationale"] else None,
            })
        return predictions

    def _evaluate_stage(self, stage_name: str, test_loader, metric_func=None):
        with torch.no_grad():
            predictions = self._predict(test_loader.dataset, f"Evaluating {stage_name}")
        with open(self._predictions_path(stage_name), "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")
        losses = [p["loss"] for p in predictions if p["loss"] is not None]
        metrics = {"test_loss": float(np.mean(losses)) if losses else None, **answer_metrics(predictions)}
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_label"}, indent=2))
        return metrics

    # GRPO

    def _train_stage(self, stage_name: str, eval_only: bool = False,
                     batch_size_override: Optional[int] = None, resume: bool = False,
                     save_every_n_steps: int = 200):
        if isinstance(self.stages[stage_name], GRPOConfig):
            return self._train_stage_grpo(stage_name, eval_only, resume, save_every_n_steps)
        return super()._train_stage(stage_name, eval_only, batch_size_override, resume, save_every_n_steps)

    def _validate(self, dataset, stage_name: str) -> float:
        with torch.no_grad():
            metrics = answer_metrics(self._predict(dataset, f"Validating {stage_name}", score=False))
        print(f"val: macro_bacc={metrics.get('macro_balanced_accuracy', 0):.4f} "
              f"reward={metrics['overall']['mean_reward']:.4f} format={metrics['overall']['format_rate']:.3f}")
        return metrics.get("macro_balanced_accuracy", 0.0)

    def _pg_backward(self, sample: dict, ids: torch.Tensor, mask: torch.Tensor,
                     advantages: torch.Tensor, config: GRPOConfig) -> float:
        """Accumulate grads of -sum(A * logp) / n_tokens over the group, chunk by chunk."""
        n_tokens = mask.sum().clamp(min=1)
        total = 0.0
        for s in range(0, ids.size(0), config.pg_chunk_size):
            chunk = slice(s, s + config.pg_chunk_size)
            logp = token_logprobs(self.model, sample, ids[chunk], mask[chunk], config.temperature)
            loss = -(logp * advantages[chunk, None]).sum() / n_tokens
            (loss / config.grad_accumulation_steps).backward()
            total += loss.item()
        return total

    def _train_stage_grpo(self, stage_name: str, eval_only: bool, resume: bool, save_every_n_steps: int):
        config: GRPOConfig = self.stages[stage_name]
        print(f"\n{'=' * 60}\nStage: {stage_name} (GRPO)\n"
              f"Epochs: {config.num_epochs}, K={config.num_samples}, T={config.temperature}, "
              f"max_new_tokens={self.max_new_tokens}, grad_accum={config.grad_accumulation_steps}\n{'=' * 60}")
        torch.manual_seed(self.seed)

        self.model = self._create_model()
        self._enable_lora_if_needed(stage_name)
        if not resume:
            self._load_previous_stage_model(stage_name)

        kwargs = {"EOS_TOKEN": self.model.get_eos_token(), **config.dataset_kwargs}
        train_ds, val_ds, test_ds = (config.dataset_class(split=s, **kwargs) for s in ("train", "validation", "test"))
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=list)

        if eval_only:
            if self._load_checkpoint(stage_name, eval_only=True) is None:
                print(f"Warning: No checkpoint for {stage_name} — cannot evaluate")
                return {}
            metrics = self._evaluate_stage(stage_name, test_loader)
            self._mark_stage_completed(stage_name, metrics)
            return metrics

        optimizer = self._get_optimizer(config)
        total_steps = math.ceil(len(train_ds) / config.grad_accumulation_steps) * config.num_epochs
        scheduler = LambdaLR(optimizer, lambda s: max(0.0, 1.0 - s / max(total_steps, 1)))
        trainable = [p for p in self.model.parameters() if p.requires_grad]

        state_path = self._checkpoint_dir(stage_name) / "grpo_state.json"
        state = {"epoch": 0, "step": -1, "opt_step": 0, "best": -math.inf}
        if resume and self._load_latest_checkpoint(stage_name, optimizer, scheduler) is not None:
            state = json.loads(state_path.read_text())
        log_path = self._results_dir(stage_name) / "grpo_log.jsonl"
        eos_id = self.model.tokenizer.eos_token_id

        def maybe_save_best():
            score = self._validate(val_ds, stage_name)
            if score > state["best"]:
                state["best"] = score
                self._save_checkpoint(stage_name, state["epoch"], -score, optimizer, scheduler)
            self.model.train()

        for epoch in range(state["epoch"], config.num_epochs):
            state["epoch"] = epoch
            order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
            skip_until = state["step"]  # last finished step when resuming mid-epoch, else -1
            stats = defaultdict(list)
            accum = 0
            optimizer.zero_grad(set_to_none=True)
            for step, idx in enumerate(tqdm(order, desc=f"GRPO epoch {epoch}", dynamic_ncols=True)):
                if step <= skip_until:
                    continue
                sample = train_ds[idx]
                _prep(sample)
                self.model.eval()
                ids = generate(self.model, sample, self.max_new_tokens, config.num_samples, config.temperature)
                mask = completion_mask(ids, eos_id)
                texts = self.model.tokenizer.batch_decode(ids, skip_special_tokens=True)
                rewards = torch.tensor([train_ds.reward(sample, t) for t in texts], device=ids.device)
                stats["reward"].append(rewards.mean().item())
                stats["correct"].append(np.mean([train_ds.evaluate(sample, t) for t in texts]))
                stats["format"].append(np.mean([train_ds.is_well_formed(t) for t in texts]))
                stats["length"].append(mask.sum(1).float().mean().item())
                advantages = rewards - rewards.mean()
                if advantages.abs().max() < 1e-6:
                    stats["skipped"].append(1.0)
                    continue
                stats["skipped"].append(0.0)

                self.model.train()
                stats["loss"].append(self._pg_backward(sample, ids, mask, advantages, config))
                accum += 1
                last = step == len(order) - 1
                if accum == config.grad_accumulation_steps or (last and accum):
                    torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    state["opt_step"] += 1
                    entry = {"epoch": epoch, "step": step, "opt_step": state["opt_step"],
                             "lr": scheduler.get_last_lr()[0], "example": texts[0][:400],
                             **{k: float(np.mean(v)) for k, v in stats.items()}}
                    with open(log_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    print(f"step {state['opt_step']}: reward={entry['reward']:.3f} correct={entry['correct']:.3f} "
                          f"format={entry['format']:.3f} len={entry['length']:.0f} skipped={entry['skipped']:.2f}")
                    stats = defaultdict(list)
                    if config.eval_every and state["opt_step"] % config.eval_every == 0:
                        maybe_save_best()
                    if state["opt_step"] % save_every_n_steps == 0:
                        state["step"] = step
                        self._save_latest_checkpoint(stage_name, epoch, step, optimizer, scheduler, 0.0)
                        state_path.write_text(json.dumps(state))
            if accum:  # trailing skipped groups after the last update
                optimizer.zero_grad(set_to_none=True)
            maybe_save_best()
            state.update(epoch=epoch + 1, step=-1)
            self._save_latest_checkpoint(stage_name, epoch + 1, -1, optimizer, scheduler, 0.0)
            state_path.write_text(json.dumps(state))

        self._load_checkpoint(stage_name, eval_only=True)
        metrics = self._evaluate_stage(stage_name, test_loader)
        metrics["best_val_macro_balanced_accuracy"] = state["best"]
        self._mark_stage_completed(stage_name, metrics)
        return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stages", nargs="+", default=["rationale_sft", "rationale_grpo"],
                   choices=["rationale_sft", "rationale_grpo"])
    p.add_argument("--sft_file", default="rationale_sft_v4.arrow")
    p.add_argument("--grpo_file", default="rationale_grpo_v1.arrow")
    p.add_argument("--llm_id", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--results_dir", default="results/rationale")
    p.add_argument("--device", default=None)
    p.add_argument("--initial_checkpoint", default=None,
                   help="Warm start for the first selected stage, e.g. an SFT best_model.pt when running GRPO alone.")
    p.add_argument("--sft_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grpo_epochs", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--pg_chunk", type=int, default=2)
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--grpo_lr_encoder", type=float, default=2e-4)
    p.add_argument("--grpo_lr_projector", type=float, default=1e-4, help="Also used for LoRA.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="Rows per split, for smoke tests.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_every_n_steps", type=int, default=200,
                   help="SFT: batches between latest.pt saves; GRPO: optimizer steps.")
    args = p.parse_args()

    stages = OrderedDict({
        "rationale_sft": StageConfig(
            dataset_class=MHCRationaleQADataset, num_epochs=args.sft_epochs, lora=True,
            dataset_kwargs={"rationale_file": args.sft_file, "limit": args.limit},
        ),
        "rationale_grpo": GRPOConfig(
            dataset_class=MHCRationaleQADataset, num_epochs=args.grpo_epochs, lora=True,
            lr_encoder=args.grpo_lr_encoder, lr_projector=args.grpo_lr_projector,
            dataset_kwargs={"rationale_file": args.grpo_file, "limit": args.limit},
            num_samples=args.num_samples, temperature=args.temperature,
            grad_accumulation_steps=args.grad_accum, pg_chunk_size=args.pg_chunk, eval_every=args.eval_every,
        ),
    })
    trainer = RationaleTrainer(
        model_type="OpenTSLMSP", stages=stages, llm_id=args.llm_id, device=args.device,
        gradient_checkpointing=args.gradient_checkpointing, results_dir=args.results_dir,
        initial_checkpoint=args.initial_checkpoint, max_new_tokens=args.max_new_tokens, seed=args.seed,
    )
    trainer.run_curriculum(stages=args.stages, batch_size=args.batch_size, eval_only=args.eval_only,
                           resume=args.resume, save_every_n_steps=args.save_every_n_steps)


if __name__ == "__main__":
    main()
