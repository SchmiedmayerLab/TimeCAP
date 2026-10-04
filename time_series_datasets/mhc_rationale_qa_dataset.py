#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Structured clinical rationale QA dataset over weekly MHC windows.

Reads the arrow file written by ``scripts/export_rationale_sft.py``. The target is the
full rationale (``<findings>``, ``<reasoning>``, ``<answer>yes|no</answer>``), so SFT
teaches the format and ``evaluate`` scores the final answer for GRPO.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pyarrow as pa
import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset

from mhc.constants import CONTINUOUS_CHANNELS
from time_series_datasets.mhc_base_qa_dataset import MHCBaseQADataset

DISPLAY = {
    "cardiovascular_disease": "cardiovascular disease", "Hypertension": "hypertension",
    "sleep_diagnosis1": "a sleep disorder", "Diabetes": "diabetes",
    "Heart Failure or CHF": "heart failure (CHF)", "CAD": "coronary artery disease (CAD)",
    "Cerebrovascular Disease": "cerebrovascular disease (stroke)",
    "Atrial fibrillation (Afib)": "atrial fibrillation (AFib)", "PH": "pulmonary hypertension (PH)",
}
ANSWER_RE = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.IGNORECASE)


class MHCRationaleQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        rationale_file: str | Path,
    ) -> None:
        self._rationale_file = Path(rationale_file)
        super().__init__(split, EOS_TOKEN)

    def _load_splits(self) -> Tuple[List[dict], List[dict], List[dict]]:
        with pa.memory_map(str(self._rationale_file), "r") as src:
            rows = pa.ipc.open_file(src).read_all().to_pylist()
        return tuple([r for r in rows if r["sft_split"] == s] for s in ("train", "validation", "test"))

    def _get_answer(self, row) -> str:
        return row["rationale"]

    def _get_pre_prompt(self, row) -> str:
        age = f"{row['age']:.0f} years" if row["age"] is not None else "unknown"
        bmi = f"{row['bmi']:.1f}" if row["bmi"] is not None else "unknown"
        return (
            "You are a clinician reviewing one week of wearable sensor data at hourly resolution "
            f"(168 hours starting {row['date'][:10]}). Participant: age {age}, "
            f"sex {row['sex'] or 'unknown'}, BMI {bmi}."
        )

    def _get_post_prompt(self, row) -> str:
        condition = DISPLAY.get(row["label"], row["label"])
        return (
            f"Does this participant have {condition}? List the clinical findings in <findings>, "
            "each stating whether it supports or argues against the condition and how strongly "
            "(strong, moderate or weak). Then weigh them in <reasoning> and end with "
            "<answer>yes</answer> or <answer>no</answer>."
        )

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        values = np.asarray(row["values"], dtype=np.float32)
        prompts = []
        for idx, name in enumerate(row["channel_names"]):
            channel = torch.from_numpy(values[idx])
            display, unit = row["display_names"][idx], row["units"][idx]
            if name in CONTINUOUS_CHANNELS:
                prompts.append(MHCBaseQADataset._continuous_prompt(channel, display, f" in {unit}" if unit else ""))
            else:
                text = f"The following is the hourly fraction of time spent {display} (0 = none or nonwear):"
                prompts.append(TextTimeSeriesPrompt(text, torch.nan_to_num(channel, nan=0.0).numpy()))
        return prompts

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample.update(label=row["label"], gold=row["answer"], sample_id=row["sample_id"])
        return sample

    @staticmethod
    def extract_answer(text: str) -> str | None:
        found = ANSWER_RE.findall(text)
        return found[-1].lower() if found else None

    def evaluate(self, sample: dict, generated: str) -> bool:
        return self.extract_answer(generated) == sample["gold"]
