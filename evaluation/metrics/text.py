#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Text generation metrics — stub, pending PI alignment.

Candidates under consideration:
  - ROUGE-L: fast, interpretable, no extra dependencies.
  - BERTScore: semantic similarity, requires sentence-transformers.
  - LLM-as-judge: most flexible, requires API calls and a rubric.

Design questions to resolve before implementing:
  - Should we score statistical / structural / semantic captions separately?
  - What is the reference — pipeline captions, or human annotations?
  - How to handle multi-sentence captions with partial coverage?
"""
from __future__ import annotations


def rouge_l(references: list[str], predictions: list[str]) -> dict[str, float]:
    raise NotImplementedError("rouge_l is not yet implemented.")


def bertscore(references: list[str], predictions: list[str]) -> dict[str, float]:
    raise NotImplementedError("bertscore is not yet implemented.")
