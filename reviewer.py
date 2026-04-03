#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from models.base import BaseModel
from runtime import Annotation, CaptionResult


EVALUATE_PROMPT = """\
You are evaluating a caption written about the sensor signal shown in the plot.

Rate this caption on a scale from 0.0 to 1.0 based on:
- Accuracy: Does it correctly describe patterns visible in the plot?
- Completeness: Does it mention the key features (trends, spikes, flat regions, missing data)?

Respond with JSON only: {"score": <float>, "feedback": "<one sentence>"}"""


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

    def refine(self, result: CaptionResult) -> list[Annotation]:
        """Identify missing observations and add new captions."""
        raise NotImplementedError

    @staticmethod
    def _parse_score(text: str) -> EvaluationScore:
        """Extract score and feedback from model JSON response."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return EvaluationScore(
                    score=float(data["score"]),
                    feedback=str(data.get("feedback", "")),
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return EvaluationScore(score=0.0, feedback=f"Failed to parse: {text[:100]}")
