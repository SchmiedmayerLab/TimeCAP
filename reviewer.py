#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import re
from dataclasses import dataclass, field

from models.base import BaseModel
from timef.schema import CaptionResult


EVALUATE_PROMPT = """\
You are reviewing captions generated for sensor signals shown in the plots.

Your job is to find patterns visible in the data that are NOT covered by the existing captions. \
List every missing observation you find, organized by category:

- **statistical**: summary statistics that are wrong or missing (mean, min, max, range, variance). \
Reference exact values and the exact minute range (e.g. "mean heart rate is 72 bpm between minute 340 and minute 585").
- **structural**: specifically check for spikes, drops, and trends. \
Each must reference the exact start and end minute (e.g. "spike in step count from minute 412 to minute 418").
- **semantic**: applies ONLY to the 12 binary channels (sleep:asleep, sleep:inbed, and the 10 workout:* channels). \
Check whether active windows in these channels are described (e.g. "walking session from minute 120 to minute 180 not mentioned").

End your response with a score on its own line in the format:
SCORE: <0-100>

100 = captions fully cover all visible patterns. Lower scores = more patterns missed."""


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
                    prompt=EVALUATE_PROMPT, signals=[signal], annotations=annotations,
                )
                scores.append(self._parse_score(response.text))
        else:
            for row in result.iter_rows():
                signals = list(row.iter_channels())
                annotations = row.annotations
                if not annotations:
                    continue
                response = self.model.process(
                    prompt=EVALUATE_PROMPT,
                    signals=signals,
                    annotations=annotations,
                    multi_channel=True,
                )
                scores.append(self._parse_score(response.text))

        return EvaluationResult(scores=scores)

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
