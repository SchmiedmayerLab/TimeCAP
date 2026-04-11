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

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
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
                and self._predictions_path(stage_name).exists()
            )
        except (json.JSONDecodeError, IOError):
            return False

    def _mark_stage_completed(self, stage_name: str, metrics: dict):
        metrics["completed"] = True
        with open(self._metrics_path(stage_name), "w") as f:
            json.dump(metrics, f, indent=2)
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

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._apply_checkpoint_to_model(checkpoint)

        print(f"Loaded checkpoint from {stage_name} (epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")
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

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._apply_checkpoint_to_model(checkpoint)

        print(f"Loaded previous stage model from {prev_stage} (epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")
        return checkpoint

    def _load_initial_checkpoint(self):
        path = Path(self.initial_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
            for batch in test_loader:
                extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=PATCH_SIZE
                )
                loss = self.model.compute_loss(batch)
                total_loss += loss.item()

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
        self._load_previous_stage_model(stage_name)
        self._enable_lora_if_needed(stage_name)

        eos_token = self.model.get_eos_token()

        dataset_kwargs = {"EOS_TOKEN": eos_token, **config.dataset_kwargs}
        train_dataset = config.dataset_class(split="train", **dataset_kwargs)
        val_dataset = config.dataset_class(split="validation", **dataset_kwargs)
        test_dataset = config.dataset_class(split="test", **dataset_kwargs)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=list
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=list
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=list
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

        best_val_loss = float("inf")

        for epoch in range(config.num_epochs):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=PATCH_SIZE
                )
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

            train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    extend_time_series_to_match_patch_size_and_aggregate(
                        batch, patch_size=PATCH_SIZE
                    )
                    loss = self.model.compute_loss(batch)
                    total_val_loss += loss.item()

            val_loss = total_val_loss / len(val_loader)

            self._save_loss_history(stage_name, epoch, train_loss, val_loss)
            print(f"Epoch {epoch}/{config.num_epochs - 1} — train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
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
    )


if __name__ == "__main__":
    main()
