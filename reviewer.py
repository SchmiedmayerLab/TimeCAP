#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import re
from dataclasses import dataclass, field

from models.base import BaseModel
from timef.schema import Annotation, CaptionResult


EVALUATE_PROMPT_TEMPLATE = """\
You are reviewing captions generated for sensor signals shown in the plots.

Your job is to find patterns visible in the data that are NOT covered by the existing captions. \
List every missing observation you find, organized by category:

- **statistical**: summary statistics that are wrong or missing (mean, min, max, range, variance). \
Reference exact values and the exact {unit_singular} range (e.g. "mean heart rate is 72 bpm between {unit_singular} 340 and {unit_singular} 585").
- **structural**: specifically check for spikes, drops, and trends. \
Each must reference the exact start and end {unit_singular} (e.g. "spike in step count from {unit_singular} 412 to {unit_singular} 418").
- **semantic**: applies ONLY to the 12 binary channels (sleep:asleep, sleep:inbed, and the 10 workout:* channels). \
Check whether active windows in these channels are described (e.g. "walking session from {unit_singular} 120 to {unit_singular} 180 not mentioned").

End your response with a score on its own line in the format:
SCORE: <0-100>

100 = captions fully cover all visible patterns. Lower scores = more patterns missed."""


def _build_prompt(signals: list) -> str:
    unit = signals[0].metadata.get("unit_timestamp", "minutes") if signals else "minutes"
    unit_singular = unit.rstrip("s")
    return EVALUATE_PROMPT_TEMPLATE.format(unit_singular=unit_singular)


@dataclass(frozen=True)
class EvaluationScore:
    score: float
    feedback: str


@dataclass
class EvaluationResult:
    scores: list[EvaluationScore] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)


class Reviewer:
    """Reviews existing captions: evaluates quality and can add missing observations."""

    def __init__(self, model: BaseModel):
        self.model = model

    def evaluate(self, result: CaptionResult, per_channel: bool = False) -> EvaluationResult:
        """Score captions against signal plots.

        per_channel=False: one call per row — all signals + all captions.
        per_channel=True:  one call per channel — single signal + its captions.
        """
        scores: list[EvaluationScore] = []

        if per_channel:
            for signal, annotations in result.iter_channels():
                if not annotations:
                    continue
                response = self.model.process(
                    prompt=_build_prompt([signal]), signals=[signal], annotations=annotations,
                )
                scores.append(self._parse_score(response.text))
        else:
            for signals, _, annotations in result.iter_rows():
                if not annotations:
                    continue
                response = self.model.process(
                    prompt=_build_prompt(signals),
                    signals=signals,
                    annotations=annotations,
                    multi_channel=True,
                )
                scores.append(self._parse_score(response.text))

        return EvaluationResult(scores=scores)

    def refine(self, result: CaptionResult) -> list[Annotation]:
        """Identify missing observations and add new captions."""
        raise NotImplementedError

    @staticmethod
    def _parse_score(text: str) -> EvaluationScore:
        """Extract score and feedback from model response."""
        match = re.search(r"SCORE:\s*(\d+)", text)
        if match:
            score = min(int(match.group(1)), 100) / 100.0
            feedback = text[:match.start()].strip()
        else:
            score = 0.0
            feedback = text.strip()
        return EvaluationScore(score=score, feedback=feedback)
