#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Multi-stage curriculum learning trainer for SensorTSLM.

Supports both OpenTSLMSP and OpenTSLMFlamingo model types with automatic
checkpoint management, knowledge transfer between stages, and LoRA
fine-tuning for later stages.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Type

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import (
    BATCH_SIZE,
    GRAD_CLIP_NORM,
    LR_ENCODER,
    LR_PROJECTOR,
    PATCH_SIZE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

@dataclass
class StageConfig:
    """Configuration for a single curriculum learning stage."""

    dataset_class: Type[Dataset]
    num_epochs: int
    lora: bool = False
    lr_encoder: float = LR_ENCODER
    lr_projector: float = LR_PROJECTOR
    lr_base: float = LR_ENCODER
    batch_size: Optional[int] = None
    metric_func: Optional[Callable] = None
    dataset_kwargs: dict = field(default_factory=dict)


class CurriculumTrainer:
    """Multi-stage curriculum learning trainer supporting OpenTSLMSP and OpenTSLMFlamingo."""

    SUPPORTED_MODELS = {"OpenTSLMSP", "OpenTSLMFlamingo"}

    def __init__(
        self,
        model_type: str,
        stages: OrderedDict[str, StageConfig],
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: Optional[str] = None,
        gradient_checkpointing: bool = False,
        results_dir: str = "results",
        initial_checkpoint: Optional[str] = None,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")

        self.model_type = model_type
        self.stages = stages
        self.llm_id = llm_id
        self.gradient_checkpointing = gradient_checkpointing
        self.initial_checkpoint = initial_checkpoint
        self.model = None

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        llm_name = llm_id.replace("/", "_")
        self.base_dir = Path(results_dir) / llm_name / model_type
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # Model creation

    def _create_model(self):
        if self.model_type == "OpenTSLMSP":
            model = OpenTSLMSP(llm_id=self.llm_id, device=self.device)
        else:
            model = OpenTSLMFlamingo(
                llm_id=self.llm_id,
                device=self.device,
                cross_attn_every_n_layers=1,
            )

        if self.gradient_checkpointing and hasattr(model, "llm"):
            model.llm.gradient_checkpointing_enable()

        return model

    # Path helpers

    def _stage_dir(self, stage_name: str) -> Path:
        return self.base_dir / stage_name

    def _checkpoint_dir(self, stage_name: str) -> Path:
        path = self._stage_dir(stage_name) / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _results_dir(self, stage_name: str) -> Path:
        path = self._stage_dir(stage_name) / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _checkpoint_path(self, stage_name: str) -> Path:
        return self._checkpoint_dir(stage_name) / "best_model.pt"

    def _latest_checkpoint_path(self, stage_name: str) -> Path:
        return self._checkpoint_dir(stage_name) / "latest.pt"

    def _metrics_path(self, stage_name: str) -> Path:
        return self._results_dir(stage_name) / "metrics.json"

    def _predictions_path(self, stage_name: str) -> Path:
        return self._results_dir(stage_name) / "test_predictions.jsonl"

    # Stage completion tracking

    def _is_stage_completed(self, stage_name: str) -> bool:
        metrics_path = self._metrics_path(stage_name)
        if not metrics_path.exists():
            return False
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            return (
                metrics.get("completed", False)
                and "test_loss" in metrics
                and self._checkpoint_path(stage_name).exists()
                and self._predictions_path(stage_name).exists()
            )
        except (json.JSONDecodeError, IOError):
            return False

    def _mark_stage_completed(self, stage_name: str, metrics: dict):
        metrics_path = self._metrics_path(stage_name)
        try:
            with open(metrics_path) as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {}
        existing.update(metrics)
        existing["completed"] = True
        with open(metrics_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Stage {stage_name} marked as completed")

    # Checkpoint management

    def _save_checkpoint(
        self, stage_name: str, epoch: int, val_loss: float, optimizer, scheduler
    ):
        checkpoint: dict[str, Any] = {"epoch": epoch, "val_loss": val_loss}

        if self.model_type == "OpenTSLMSP":
            checkpoint["encoder_state"] = self.model.encoder.state_dict()
            checkpoint["projector_state"] = self.model.projector.state_dict()
            self.model.save_lora_state_to_checkpoint(checkpoint)
        else:
            checkpoint["model_state"] = self.model.model.state_dict()

        checkpoint["optimizer_state"] = optimizer.state_dict()
        checkpoint["scheduler_state"] = scheduler.state_dict()

        torch.save(checkpoint, self._checkpoint_path(stage_name))
        print(f"Saved checkpoint at epoch {epoch} (val_loss={val_loss:.4f})")

    def _load_checkpoint(self, stage_name: str, eval_only: bool = False):
        path = self._checkpoint_path(stage_name)
        if not path.exists():
            return None

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self._apply_checkpoint_to_model(checkpoint)

        print(f"Loaded checkpoint from {stage_name} (epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")
        return checkpoint

    def _save_latest_checkpoint(
        self,
        stage_name: str,
        epoch: int,
        step: int,
        optimizer,
        scheduler,
        total_train_loss: float,
    ):
        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "total_train_loss": total_train_loss,
        }

        if self.model_type == "OpenTSLMSP":
            checkpoint["encoder_state"] = self.model.encoder.state_dict()
            checkpoint["projector_state"] = self.model.projector.state_dict()
            self.model.save_lora_state_to_checkpoint(checkpoint)
        else:
            checkpoint["model_state"] = self.model.model.state_dict()

        checkpoint["optimizer_state"] = optimizer.state_dict()
        checkpoint["scheduler_state"] = scheduler.state_dict()

        tmp_path = self._latest_checkpoint_path(stage_name).with_suffix(".pt.tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(self._latest_checkpoint_path(stage_name))

    def _load_latest_checkpoint(
        self, stage_name: str, optimizer, scheduler
    ):
        path = self._latest_checkpoint_path(stage_name)
        if not path.exists():
            return None

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self._apply_checkpoint_to_model(checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(
            f"Resuming {stage_name} from latest.pt "
            f"(epoch={checkpoint['epoch']}, step={checkpoint['step']})"
        )
        return checkpoint

    def _load_previous_stage_model(self, stage_name: str):
        stage_names = list(self.stages.keys())
        idx = stage_names.index(stage_name)

        if idx == 0:
            if self.initial_checkpoint:
                return self._load_initial_checkpoint()
            print("First stage — no previous model to load")
            return None

        prev_stage = stage_names[idx - 1]
        path = self._checkpoint_path(prev_stage)

        if not path.exists():
            print(f"Warning: No checkpoint found for previous stage {prev_stage}")
            return None

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self._apply_checkpoint_to_model(checkpoint)

        print(f"Loaded previous stage model from {prev_stage} (epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")
        return checkpoint

    def _load_initial_checkpoint(self):
        path = Path(self.initial_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self._apply_checkpoint_to_model(checkpoint)

        print(f"Loaded initial checkpoint from {path}")
        return checkpoint

    def _apply_checkpoint_to_model(self, checkpoint: dict):
        if self.model_type == "OpenTSLMSP":
            self.model.encoder.load_state_dict(checkpoint["encoder_state"])
            self.model.projector.load_state_dict(checkpoint["projector_state"])
            self.model.load_lora_state_from_checkpoint(
                checkpoint, allow_missing=True
            )
        else:
            self.model.model.load_state_dict(
                checkpoint["model_state"], strict=False
            )

    # LoRA

    def _enable_lora_if_needed(self, stage_name: str):
        config = self.stages[stage_name]
        if not config.lora:
            return
        if self.model_type != "OpenTSLMSP":
            print("LoRA not supported for Flamingo — skipping")
            return
        self.model.enable_lora(lora_r=16, lora_alpha=32, lora_dropout=0.0)
        print(f"LoRA enabled for stage {stage_name}")

    # Optimizer

    def _get_optimizer(self, config: StageConfig):
        if self.model_type == "OpenTSLMSP":
            param_groups = [
                {
                    "params": list(self.model.encoder.parameters()),
                    "lr": config.lr_encoder,
                    "weight_decay": WEIGHT_DECAY,
                },
                {
                    "params": list(self.model.projector.parameters()),
                    "lr": config.lr_projector,
                    "weight_decay": WEIGHT_DECAY,
                },
            ]
            if config.lora:
                lora_params = list(self.model.get_lora_parameters())
                if lora_params:
                    param_groups.append(
                        {
                            "params": lora_params,
                            "lr": config.lr_projector,
                            "weight_decay": WEIGHT_DECAY,
                        }
                    )
        else:
            params_with_wd = []
            params_without_wd = []
            for name, param in self.model.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "gated_cross_attn" in name:
                    params_with_wd.append(param)
                else:
                    params_without_wd.append(param)
            param_groups = [
                {"params": params_with_wd, "weight_decay": 0.1},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]

        return AdamW(param_groups, lr=config.lr_base)

    # Debug logging

    @staticmethod
    def _log_sample_input(sample: dict, stage_name: str):
        """Print a single training sample to verify inputs are formed correctly."""
        print(f"\n{'-' * 60}")
        print(f"Sample input for stage {stage_name}")
        print(f"{'-' * 60}")
        print(f"[pre_prompt]\n{sample.get('pre_prompt', '')}")
        ts_texts = sample.get("time_series_text") or []
        ts_values = sample.get("time_series") if sample.get("time_series") is not None else []
        for i, (text, ts) in enumerate(zip(ts_texts, ts_values)):
            arr = np.asarray(ts).ravel()
            preview = ", ".join(f"{v:.3f}" for v in arr[:10].tolist())
            print(
                f"[ts {i}] {text}\n  len={len(arr)}, first10=[{preview}]"
            )
        print(f"[post_prompt]\n{sample.get('post_prompt', '')}")
        print(f"[answer]\n{sample.get('answer', '')}")
        print(f"{'-' * 60}\n")

    # Loss history

    def _save_loss_history(
        self, stage_name: str, epoch: int, train_loss: float, val_loss: float
    ):
        path = self._checkpoint_dir(stage_name) / "loss_history.txt"
        write_header = not path.exists()
        with open(path, "a") as f:
            if write_header:
                f.write("Epoch\tTrain_Loss\tVal_Loss\n")
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

    # Evaluation

    def _evaluate_stage(self, stage_name: str, test_loader, metric_func=None):
        self.model.eval()
        total_loss = 0.0
        predictions = []

        with torch.no_grad():
            test_pbar = tqdm(
                test_loader, desc=f"Evaluating {stage_name}", dynamic_ncols=True
            )
            for step, batch in enumerate(test_pbar):
                extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=PATCH_SIZE
                )
                loss = self.model.compute_loss(batch)
                total_loss += loss.item()
                test_pbar.set_postfix(
                    avg_loss=f"{total_loss / (step + 1):.4f}"
                )

                generated = self.model.generate(batch, max_new_tokens=2000)
                for i, gen_text in enumerate(generated):
                    predictions.append(
                        {
                            "pre_prompt": batch[i].get("pre_prompt", ""),
                            "post_prompt": batch[i].get("post_prompt", ""),
                            "generated": gen_text,
                            "gold": batch[i].get("answer", ""),
                        }
                    )

        test_loss = total_loss / len(test_loader)

        with open(self._predictions_path(stage_name), "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        metrics = {"test_loss": test_loss}
        if metric_func:
            metrics.update(metric_func(predictions))

        print(f"Evaluation complete: {metrics}")
        return metrics

    # Core training loop

    def _train_stage(
        self,
        stage_name: str,
        eval_only: bool = False,
        batch_size_override: Optional[int] = None,
        resume: bool = False,
        save_every_n_steps: int = 200,
    ):
        config = self.stages[stage_name]
        batch_size = batch_size_override or config.batch_size or BATCH_SIZE

        print(
            f"\n{'=' * 60}\n"
            f"Stage: {stage_name}\n"
            f"Epochs: {config.num_epochs}, Batch size: {batch_size}, "
            f"LoRA: {config.lora}, Model: {self.model_type}\n"
            f"{'=' * 60}"
        )

        self.model = self._create_model()
        self._enable_lora_if_needed(stage_name)
        if not resume:
            self._load_previous_stage_model(stage_name)

        eos_token = self.model.get_eos_token()

        dataset_kwargs = {"EOS_TOKEN": eos_token, **config.dataset_kwargs}
        train_dataset = config.dataset_class(split="train", **dataset_kwargs)
        val_dataset = config.dataset_class(split="validation", **dataset_kwargs)
        test_dataset = config.dataset_class(split="test", **dataset_kwargs)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=list
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, collate_fn=list
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, collate_fn=list
        )

        if eval_only:
            checkpoint = self._load_checkpoint(stage_name, eval_only=True)
            if checkpoint is None:
                print(f"Warning: No checkpoint for {stage_name} — cannot evaluate")
                return {}
            metrics = self._evaluate_stage(
                stage_name, test_loader, config.metric_func
            )
            self._mark_stage_completed(stage_name, metrics)
            return metrics

        optimizer = self._get_optimizer(config)
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        start_epoch = 0
        start_step = 0
        start_train_loss = 0.0
        if resume:
            resume_ckpt = self._load_latest_checkpoint(
                stage_name, optimizer, scheduler
            )
            if resume_ckpt is not None:
                start_epoch = resume_ckpt["epoch"]
                start_step = resume_ckpt["step"] + 1
                start_train_loss = resume_ckpt.get("total_train_loss", 0.0)

        best_val_loss = float("inf")

        for epoch in range(start_epoch, config.num_epochs):
            self.model.train()
            total_train_loss = start_train_loss if epoch == start_epoch else 0.0
            step_skip = start_step if epoch == start_epoch else 0

            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{config.num_epochs - 1} train",
                dynamic_ncols=True,
                initial=step_skip,
                total=len(train_loader),
            )
            for step, batch in enumerate(train_loader):
                if step < step_skip:
                    continue
                extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=PATCH_SIZE
                )
                if epoch == 0 and step == 0:
                    self._log_sample_input(batch[0], stage_name)
                optimizer.zero_grad()
                loss = self.model.compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    GRAD_CLIP_NORM,
                )
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                seen = step + 1 - step_skip
                train_pbar.update(1)
                train_pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg=f"{total_train_loss / max(seen, 1):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if (step + 1) % save_every_n_steps == 0:
                    self._save_latest_checkpoint(
                        stage_name,
                        epoch,
                        step,
                        optimizer,
                        scheduler,
                        total_train_loss,
                    )
            train_pbar.close()
            # reset resume state so later epochs start from step 0
            start_step = 0
            start_train_loss = 0.0

            train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                val_pbar = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch}/{config.num_epochs - 1} val",
                    dynamic_ncols=True,
                )
                for step, batch in enumerate(val_pbar):
                    extend_time_series_to_match_patch_size_and_aggregate(
                        batch, patch_size=PATCH_SIZE
                    )
                    loss = self.model.compute_loss(batch)
                    total_val_loss += loss.item()
                    val_pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        avg=f"{total_val_loss / (step + 1):.4f}",
                    )

            val_loss = total_val_loss / len(val_loader)

            self._save_loss_history(stage_name, epoch, train_loss, val_loss)
            print(f"Epoch {epoch}/{config.num_epochs - 1} — train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss + 1e-4 < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(
                    stage_name, epoch, val_loss, optimizer, scheduler
                )

        self._load_checkpoint(stage_name, eval_only=True)

        metrics = self._evaluate_stage(
            stage_name, test_loader, config.metric_func
        )
        metrics["best_val_loss"] = best_val_loss

        self._mark_stage_completed(stage_name, metrics)
        return metrics

    # Orchestration

    def run_curriculum(
        self,
        stages: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
        eval_only: bool = False,
        resume: bool = False,
        save_every_n_steps: int = 200,
    ):
        stage_names = stages or list(self.stages.keys())

        for name in stage_names:
            if name not in self.stages:
                raise ValueError(
                    f"Unknown stage: {name}. "
                    f"Available: {list(self.stages.keys())}"
                )

        results = {}

        for stage_name in stage_names:
            if self._is_stage_completed(stage_name):
                print(f"Skipping completed stage: {stage_name}")
                with open(self._metrics_path(stage_name)) as f:
                    results[stage_name] = json.load(f)
                continue

            metrics = self._train_stage(
                stage_name,
                eval_only=eval_only,
                batch_size_override=batch_size,
                resume=resume,
                save_every_n_steps=save_every_n_steps,
            )
            results[stage_name] = metrics

        results_path = self.base_dir / "curriculum_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Curriculum complete. Results saved to {results_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description="SensorTSLM Curriculum Learning")
    parser.add_argument(
        "--model",
        required=True,
        choices=["OpenTSLMSP", "OpenTSLMFlamingo"],
    )
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--initial_checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest.pt in the (first specified) stage dir.",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=200,
        help="Write latest.pt every N training steps for mid-epoch resume.",
    )
    args = parser.parse_args()

    from time_series_datasets.mhc_caption_qa_dataset import MHCCaptionQADataset

    curriculum_stages = OrderedDict(
        {
            "stage1_captioning": StageConfig(
                dataset_class=MHCCaptionQADataset,
                num_epochs=1,
                lora=False,
                dataset_kwargs={"export_dir": "exports/lean_smoke"},
            ),
            # stage2_rationale: add when dataset is ready
        }
    )

    trainer = CurriculumTrainer(
        model_type=args.model,
        stages=curriculum_stages,
        llm_id=args.llm_id,
        device=args.device,
        gradient_checkpointing=args.gradient_checkpointing,
        initial_checkpoint=args.initial_checkpoint,
    )

    trainer.run_curriculum(
        stages=args.stages,
        batch_size=args.batch_size,
        eval_only=args.eval_only,
        resume=args.resume,
        save_every_n_steps=args.save_every_n_steps,
    )


if __name__ == "__main__":
    main()
