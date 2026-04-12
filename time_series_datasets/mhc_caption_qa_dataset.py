#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Caption generation QA dataset."""

from __future__ import annotations

from time_series_datasets.mhc_base_qa_dataset import MHCBaseQADataset


class MHCCaptionQADataset(MHCBaseQADataset):
    def _get_answer(self, row) -> str:
        return row["caption"]

    def _get_post_prompt(self, _row) -> str:
        return (
            "Please generate a detailed caption for this day of sensor data, "
            "describing the person's activity, sleep, and physiological "
            "patterns as accurately as possible."
        )
