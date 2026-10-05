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
  - several prompts are sampled per ``generate`` call (left-padded, see
    models/opentslm_native._assemble_left_padded), and evaluation is batched the same way

The GRPO stage first evaluates its warm start (the SFT model) on the GRPO test split, so
every run yields a paired SFT-vs-GRPO comparison on identical test samples.

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
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from curriculum_learning import CurriculumTrainer, StageConfig
from models.opentslm_native import _assemble_left_padded
from opentslm.model_config import GRAD_CLIP_NORM, PATCH_SIZE
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from time_series_datasets.mhc_rationale_qa_dataset import MHCRationaleQADataset


@dataclass
class GRPOConfig(StageConfig):
    num_samples: int = 8
    temperature: float = 1.0
    grad_accumulation_steps: int = 8
    pg_chunk_size: int = 2
    prompts_per_generate: int = 2  # prompts sampled together; P * K sequences per generate call
    eval_every: int = 0  # optimizer steps between validations; 0 = once per epoch
    baseline_eval: bool = True  # evaluate the warm start on the test split before training


# Model helpers (OpenTSLMSP)

def _prep(samples: list[dict]) -> list[dict]:
    extend_time_series_to_match_patch_size_and_aggregate(samples, patch_size=PATCH_SIZE)
    return samples


def completion_mask(ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """1 for tokens up to and including the first EOS of each row, 0 after."""
    is_eos = ids == eos_id
    first = torch.where(is_eos.any(1), is_eos.int().argmax(1), torch.full_like(ids[:, 0], ids.size(1)))
    return (torch.arange(ids.size(1), device=ids.device)[None] <= first[:, None]).long()


def token_logprobs(model, samples: list[dict], ids: torch.Tensor, mask: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
    """Per-token log-probs [B, T] of ``ids[i]`` after the prompt of ``samples[i]``.

    A single sample is shared by all rows. Prompts are left-padded, so position ids come
    from the attention mask, matching what ``generate`` uses when it samples.
    """
    prompt, prompt_mask = _assemble_left_padded(model, samples)
    b, t = ids.shape
    if prompt.size(0) != b:
        prompt, prompt_mask = prompt.expand(b, -1, -1), prompt_mask.expand(b, -1)
    embeds = torch.cat([prompt, model.llm.get_input_embeddings()(ids)], dim=1)
    attn = torch.cat([prompt_mask, mask], dim=1)
    positions = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model.llm(inputs_embeds=embeds, attention_mask=attn, position_ids=positions,
                       logits_to_keep=t + 1, return_dict=True).logits[:, :-1].float()
    logp = torch.log_softmax(logits / temperature, dim=-1).gather(2, ids.unsqueeze(-1)).squeeze(-1)
    return logp * mask


@torch.no_grad()
def generate(model, samples: list[dict], max_new_tokens: int, num_samples: int = 1,
             temperature: Optional[float] = None) -> torch.Tensor:
    """New token ids [len(samples) * K, T], K consecutive rows per sample; greedy when
    ``temperature`` is None."""
    prompt, prompt_mask = _assemble_left_padded(model, samples)
    tok = model.tokenizer
    sampling = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if temperature else dict(do_sample=False)
    return model.llm.generate(
        inputs_embeds=prompt.repeat_interleave(num_samples, dim=0),
        attention_mask=prompt_mask.repeat_interleave(num_samples, dim=0),
        max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        **sampling,
    )


@torch.no_grad()
def answer_scores(model, samples: list[dict], texts: list[str], chunk: int = 4) -> list[float]:
    """log p(yes) - log p(no) for the answer following each sample's own rationale."""
    scores = []
    for start in range(0, len(samples), chunk):
        part, seqs = samples[start:start + chunk], []
        for text in texts[start:start + chunk]:
            cut = text.rfind("<answer>")
            prefix = text[:cut] if cut >= 0 else text.rstrip() + "\n"
            # Answers are tokenized with special tokens in SFT (compute_loss), so score them the same way.
            seqs += [torch.tensor(model.tokenizer(prefix + f"<answer>{a}</answer>").input_ids) for a in ("yes", "no")]
        ids = pad_sequence(seqs, batch_first=True, padding_value=model.tokenizer.pad_token_id).to(model.device)
        mask = pad_sequence([torch.ones_like(q) for q in seqs], batch_first=True).to(model.device)
        lp = token_logprobs(model, [s for s in part for _ in range(2)], ids, mask).sum(1)
        scores += (lp[0::2] - lp[1::2]).tolist()
    return scores


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


def _dist() -> bool:
    return dist.is_available() and dist.is_initialized()


class RationaleTrainer(CurriculumTrainer):
    def __init__(self, *args, max_new_tokens: int = 512, eval_batch_size: int = 16, seed: int = 0,
                 attn_implementation: str = "eager", **kwargs):
        super().__init__(*args, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.eval_batch_size = eval_batch_size
        self.seed = seed
        self.attn_implementation = attn_implementation
        self.rank = dist.get_rank() if _dist() else 0
        self.world = dist.get_world_size() if _dist() else 1

    def _create_model(self):
        model = super()._create_model()
        if self.attn_implementation != "eager":  # OpenTSLMSP loads eager; switch before LoRA wraps it
            model.llm.set_attn_implementation(self.attn_implementation)
        return model

    def _print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs, flush=True)

    def _mark_stage_completed(self, stage_name: str, metrics: dict):
        if self.rank == 0:
            super()._mark_stage_completed(stage_name, metrics)

    def run_curriculum(self, stages=None, batch_size=None, eval_only=False, resume=False, save_every_n_steps=200):
        if self.world == 1:
            return super().run_curriculum(stages, batch_size, eval_only, resume, save_every_n_steps)
        names = stages or list(self.stages)
        if not all(isinstance(self.stages[n], GRPOConfig) for n in names):
            raise ValueError("multi-GPU runs support the GRPO stage only; run SFT on one GPU and pass "
                             "its best_model.pt as --initial_checkpoint")
        self.stages = OrderedDict((n, self.stages[n]) for n in names)
        results = {n: self._train_stage_grpo(n, eval_only, resume, save_every_n_steps)
                   for n in names if not self._is_stage_completed(n)}
        if self.rank == 0:
            (self.base_dir / "curriculum_results.json").write_text(json.dumps(results, indent=2))
        return results

    # Evaluation (used by both stages)

    def _predict(self, dataset, desc: str, score: bool = True) -> list[dict]:
        """Greedy predictions for every row; rows are sharded across ranks and gathered."""
        self.model.eval()
        mine = list(range(self.rank, len(dataset), self.world))
        predictions = []
        for start in tqdm(range(0, len(mine), self.eval_batch_size), desc=desc, dynamic_ncols=True,
                          disable=self.rank != 0):
            idxs = mine[start:start + self.eval_batch_size]
            samples = _prep([dataset[i] for i in idxs])
            texts = self.model.tokenizer.batch_decode(
                generate(self.model, samples, self.max_new_tokens), skip_special_tokens=True)
            scores = answer_scores(self.model, samples, texts) if score else [None] * len(samples)
            for i, sample, text, sc in zip(idxs, samples, texts, scores):
                predictions.append({
                    "idx": i, "sample_id": sample["sample_id"], "label": sample["label"], "gold": sample["gold"],
                    "pred": dataset.extract_answer(text), "well_formed": dataset.is_well_formed(text),
                    "reward": dataset.reward(sample, text), "score": sc, "generated": text,
                    "loss": float(self.model.compute_loss([sample])) if sample["has_rationale"] else None,
                })
        if self.world > 1:
            gathered = [None] * self.world
            dist.all_gather_object(gathered, predictions)
            predictions = sorted((p for part in gathered for p in part), key=lambda p: p["idx"])
        return predictions

    def _evaluate_stage(self, stage_name: str, test_loader, metric_func=None, prefix: str = ""):
        with torch.no_grad():
            predictions = self._predict(test_loader.dataset, f"Evaluating {prefix}{stage_name}")
        losses = [p["loss"] for p in predictions if p["loss"] is not None]
        metrics = {"test_loss": float(np.mean(losses)) if losses else None, **answer_metrics(predictions)}
        if self.rank == 0:
            path = self._predictions_path(stage_name)
            with open(path.with_name(prefix + path.name), "w") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")
            print(json.dumps({k: v for k, v in metrics.items() if k != "per_label"}, indent=2), flush=True)
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
        self._print(f"val: macro_bacc={metrics.get('macro_balanced_accuracy', 0):.4f} "
                    f"reward={metrics['overall']['mean_reward']:.4f} format={metrics['overall']['format_rate']:.3f}")
        return metrics.get("macro_balanced_accuracy", 0.0)

    def _pg_backward(self, sample: dict, ids: torch.Tensor, mask: torch.Tensor,
                     advantages: torch.Tensor, config: GRPOConfig) -> float:
        """Accumulate grads of -sum(A * logp) / n_tokens over one prompt's group, chunk by chunk.

        Unscaled: the optimizer step divides the summed grads by the number of contributing
        prompts across all ranks.
        """
        n_tokens = mask.sum().clamp(min=1)
        total = 0.0
        for s in range(0, ids.size(0), config.pg_chunk_size):
            chunk = slice(s, s + config.pg_chunk_size)
            logp = token_logprobs(self.model, [sample], ids[chunk], mask[chunk], config.temperature)
            loss = -(logp * advantages[chunk, None]).sum() / n_tokens
            loss.backward()
            total += loss.item()
        return total

    def _train_stage_grpo(self, stage_name: str, eval_only: bool, resume: bool, save_every_n_steps: int):
        config: GRPOConfig = self.stages[stage_name]
        P, K, A = config.prompts_per_generate, config.num_samples, config.grad_accumulation_steps
        self._print(f"\n{'=' * 60}\nStage: {stage_name} (GRPO)\n"
                    f"Epochs: {config.num_epochs}, K={K}, T={config.temperature}, max_new_tokens={self.max_new_tokens}, "
                    f"GPUs={self.world}, prompts/generate={P}, prompts per GPU per step={A}, "
                    f"global batch={A * self.world} prompts, attention={self.attn_implementation}\n{'=' * 60}")
        torch.manual_seed(self.seed)  # identical model init on every rank

        self.model = self._create_model()
        self._enable_lora_if_needed(stage_name)

        kwargs = {"EOS_TOKEN": self.model.get_eos_token(), **config.dataset_kwargs}
        train_ds, val_ds, test_ds = (config.dataset_class(split=s, **kwargs) for s in ("train", "validation", "test"))
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=list)

        # Every rank walks the same shuffled order and takes every world-th group of P prompts;
        # a step is A prompts per rank, so all ranks reach each all-reduce together.
        per_rank = (len(train_ds) // (P * self.world)) * P
        steps_per_epoch = math.ceil(per_rank / A)
        optimizer = self._get_optimizer(config)
        total_steps = steps_per_epoch * config.num_epochs
        scheduler = LambdaLR(optimizer, lambda s: max(0.0, 1.0 - s / max(total_steps, 1)))
        state_path = self._checkpoint_dir(stage_name) / "grpo_state.json"
        state = {"epoch": 0, "step": -1, "opt_step": 0, "best": -math.inf}
        resumed = resume and state_path.exists() and \
            self._load_latest_checkpoint(stage_name, optimizer, scheduler) is not None
        if resumed:
            state = json.loads(state_path.read_text())
        elif not eval_only:
            self._load_previous_stage_model(stage_name)  # SFT warm start

        if eval_only:
            if self._load_checkpoint(stage_name, eval_only=True) is None:
                self._print(f"Warning: No checkpoint for {stage_name} — cannot evaluate")
                return {}
            metrics = self._evaluate_stage(stage_name, test_loader)
            self._mark_stage_completed(stage_name, metrics)
            return metrics

        baseline_path = self._results_dir(stage_name) / "baseline_metrics.json"
        if config.baseline_eval and not resumed and not baseline_path.exists():
            baseline = self._evaluate_stage(stage_name, test_loader, prefix="baseline_")
            if self.rank == 0:
                baseline_path.write_text(json.dumps(baseline, indent=2))

        torch.manual_seed(self.seed * 1000 + self.rank + 1)  # different completions per rank
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        log_path = self._results_dir(stage_name) / "grpo_log.jsonl"
        eos_id = self.model.tokenizer.eos_token_id
        device = next(iter(trainable)).device
        self._print(f"{len(train_ds)} train prompts -> {per_rank} per GPU, {steps_per_epoch} steps per epoch")

        def save_latest(step: int):
            state["step"] = step
            if self.rank == 0:
                self._save_latest_checkpoint(stage_name, state["epoch"], step, optimizer, scheduler, 0.0)
                state_path.write_text(json.dumps(state))

        def maybe_save_best():
            score = self._validate(val_ds, stage_name)
            if score > state["best"]:
                state["best"] = score
                if self.rank == 0:
                    self._save_checkpoint(stage_name, state["epoch"], -score, optimizer, scheduler)
            if _dist():
                dist.barrier()

        # per-step sums: reward, correct, format, length, prompts, skipped, loss
        sums = torch.zeros(7, dtype=torch.float64, device=device)
        accum, last_text = 0, ""

        def optimizer_step(step: int):
            nonlocal accum, sums
            contributing = sums[4] - sums[5]
            if _dist():
                flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in trainable])
                dist.all_reduce(flat)
                dist.all_reduce(sums)
                contributing = sums[4] - sums[5]
                offset = 0
                for p in trainable:
                    n = p.numel()
                    p.grad = flat[offset:offset + n].view_as(p).clone()
                    offset += n
            if contributing > 0:
                for p in trainable:
                    if p.grad is not None:
                        p.grad /= contributing
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["opt_step"] += 1
            n = max(sums[4].item(), 1)
            entry = {"epoch": state["epoch"], "step": step, "opt_step": state["opt_step"],
                     "lr": scheduler.get_last_lr()[0], "prompts": int(sums[4].item()),
                     "reward": sums[0].item() / n, "correct": sums[1].item() / n, "format": sums[2].item() / n,
                     "length": sums[3].item() / n, "skipped": sums[5].item() / n,
                     "loss": sums[6].item() / max(contributing.item(), 1), "example": last_text[:400]}
            if self.rank == 0:
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                print(f"step {state['opt_step']}/{total_steps}: reward={entry['reward']:.3f} correct={entry['correct']:.3f} "
                      f"format={entry['format']:.3f} len={entry['length']:.0f} skipped={entry['skipped']:.2f}", flush=True)
            sums = torch.zeros_like(sums)
            accum = 0
            if config.eval_every and state["opt_step"] % config.eval_every == 0:
                maybe_save_best()
            if state["opt_step"] % save_every_n_steps == 0:
                save_latest(step)

        for epoch in range(state["epoch"], config.num_epochs):
            state["epoch"] = epoch
            order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
            groups = [order[g:g + P] for g in range(0, per_rank * self.world, P)][self.rank::self.world]
            skip_until = state["step"]  # last finished prompt group when resuming mid-epoch, else -1
            optimizer.zero_grad(set_to_none=True)
            for step, idxs in enumerate(tqdm(groups, desc=f"GRPO epoch {epoch}", dynamic_ncols=True,
                                             disable=self.rank != 0)):
                if step <= skip_until:
                    continue
                samples = _prep([train_ds[i] for i in idxs])
                self.model.eval()
                ids = generate(self.model, samples, self.max_new_tokens, K, config.temperature)
                mask = completion_mask(ids, eos_id)
                texts = self.model.tokenizer.batch_decode(ids, skip_special_tokens=True)
                last_text = texts[0]
                for j, sample in enumerate(samples):
                    rows = slice(j * K, (j + 1) * K)
                    group = texts[rows]
                    rewards = torch.tensor([train_ds.reward(sample, t) for t in group], device=ids.device)
                    advantages = rewards - rewards.mean()
                    skipped = advantages.abs().max() < 1e-6
                    sums += torch.tensor([
                        rewards.mean().item(), np.mean([train_ds.evaluate(sample, t) for t in group]),
                        np.mean([train_ds.is_well_formed(t) for t in group]), mask[rows].sum(1).float().mean().item(),
                        1.0, float(skipped), 0.0], dtype=torch.float64, device=device)
                    if not skipped:
                        self.model.train()
                        sums[6] += self._pg_backward(sample, ids[rows], mask[rows], advantages, config)
                    accum += 1
                    if accum == A:
                        optimizer_step(step)
            if accum:  # leftover prompts; equal on every rank since groups are equal
                optimizer_step(len(groups) - 1)
            maybe_save_best()
            state["epoch"] = epoch + 1
            save_latest(-1)

        if _dist():
            dist.barrier()
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
    p.add_argument("--pg_chunk", type=int, default=2, help="Completions per backward pass.")
    p.add_argument("--prompts_per_generate", type=int, default=2, help="GRPO prompts sampled per generate call.")
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--no_baseline_eval", action="store_true", help="Skip evaluating the SFT warm start on the GRPO test split.")
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--grpo_lr_encoder", type=float, default=2e-4)
    p.add_argument("--grpo_lr_projector", type=float, default=1e-4, help="Also used for LoRA.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                   help="LLM attention; sdpa avoids materialising L x L attention for long prompts.")
    p.add_argument("--limit", type=int, default=None, help="Rows per split, for smoke tests.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_every_n_steps", type=int, default=200,
                   help="SFT: batches between latest.pt saves; GRPO: optimizer steps.")
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:  # launched by torchrun: one process per GPU
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
            prompts_per_generate=args.prompts_per_generate, baseline_eval=not args.no_baseline_eval,
        ),
    })
    trainer = RationaleTrainer(
        model_type="OpenTSLMSP", stages=stages, llm_id=args.llm_id, device=args.device,
        gradient_checkpointing=args.gradient_checkpointing, results_dir=args.results_dir,
        initial_checkpoint=args.initial_checkpoint, max_new_tokens=args.max_new_tokens,
        eval_batch_size=args.eval_batch_size, seed=args.seed, attn_implementation=args.attn,
    )
    trainer.run_curriculum(stages=args.stages, batch_size=args.batch_size, eval_only=args.eval_only,
                           resume=args.resume, save_every_n_steps=args.save_every_n_steps)


if __name__ == "__main__":
    main()
    if _dist():
        dist.destroy_process_group()
