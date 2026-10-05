#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Structured clinical rationale QA dataset over weekly MHC windows.

Reads the arrow files written by ``scripts/export_rationale_sft.py`` (rows carry a full
rationale: ``<findings>``, ``<reasoning>``, ``<answer>yes|no</answer>``) and
``scripts/export_rationale_grpo.py`` (label-only rows, rationale is null). SFT trains on
the rationale; GRPO only needs the label, which ``reward`` checks.

Bypasses ``QADataset``'s class-level cache so SFT and GRPO files can be loaded in the
same process. Splits are kept as Arrow tables (cached per file and limit) and rows are
formatted on access, so a 100k-row GRPO file loads quickly in every training process.
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
SPLITS = ("train", "validation", "test")
ANSWER_RE = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.IGNORECASE)
_BODY = r"(?:(?!</?(?:findings|reasoning|answer)>).)"  # any char that does not open a block tag
FORMAT_RE = re.compile(
    rf"\s*<findings>{_BODY}*?\S{_BODY}*?</findings>\s*<reasoning>{_BODY}*?\S{_BODY}*?</reasoning>\s*"
    r"<answer>\s*(?:yes|no)\s*</answer>\s*",
    re.IGNORECASE | re.DOTALL,
)

_CACHE: dict[tuple, tuple[pa.Table, pa.Table, pa.Table]] = {}


class MHCRationaleQADataset(QADataset):
    # Reward for a well-formed rationale with the wrong answer; a correct one scores 1.0
    # and anything malformed 0.0, so the answer only counts when the format holds.
    FORMAT_REWARD = 0.1

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        rationale_file: str | Path,
        limit: int | None = None,
    ) -> None:
        self.EOS_TOKEN = EOS_TOKEN
        self._rationale_file = Path(rationale_file)
        key = (str(self._rationale_file.resolve()), limit)
        if key not in _CACHE:
            _CACHE[key] = tuple(t.slice(0, limit) for t in self._load_splits())
        self._table = _CACHE[key][SPLITS.index(split)]

    def _load_splits(self) -> Tuple[pa.Table, pa.Table, pa.Table]:
        table = pa.ipc.open_file(pa.memory_map(str(self._rationale_file), "r")).read_all()
        return tuple(table.filter(pa.compute.equal(table["split"], s)) for s in SPLITS)

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, idx: int) -> dict:
        if not 0 <= idx < self._table.num_rows:
            raise IndexError(idx)
        return self._format_sample(self._table.slice(idx, 1).to_pylist()[0])

    def _get_answer(self, row) -> str:
        return row["rationale"] or f"<answer>{row['answer']}</answer>"

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
        sample.update(label=row["label"], gold=row["answer"], sample_id=row["sample_id"],
                      has_rationale=row["rationale"] is not None)
        return sample

    @staticmethod
    def extract_answer(text: str) -> str | None:
        found = ANSWER_RE.findall(text)
        return found[-1].lower() if found else None

    @staticmethod
    def is_well_formed(text: str) -> bool:
        """Exactly findings, then reasoning, then one answer, with nothing after it."""
        return FORMAT_RE.fullmatch(text) is not None

    def evaluate(self, sample: dict, generated: str) -> bool:
        return self.extract_answer(generated) == sample["gold"]

    def reward(self, sample: dict, generated: str) -> float:
        if not self.is_well_formed(generated):
            return 0.0
        return 1.0 if self.evaluate(sample, generated) else self.FORMAT_REWARD
