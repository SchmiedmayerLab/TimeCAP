#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

LABEL_SCHEMAS: dict[str, dict[str, list[str] | str]] = {
    "har_uci": {
        "description": "UCI HAR Dataset activity labels",
        "labels": [
            "walking",
            "walking_upstairs",
            "walking_downstairs",
            "sitting",
            "standing",
            "laying",
        ],
    },
    "aasm": {
        "description": "AASM sleep staging (American Academy of Sleep Medicine)",
        "labels": ["W", "N1", "N2", "N3", "REM"],
    },
    "aasm_verbose": {
        "description": "AASM sleep staging with verbose labels",
        "labels": ["Wake", "Non-REM stage 1", "Non-REM stage 2", "Non-REM stage 3", "REM"],
    },
    "binary": {
        "description": "Binary yes/no classification",
        "labels": ["yes", "no"],
    },
    "ecg_rhythm": {
        "description": "Basic ECG rhythm classifications",
        "labels": [
            "normal_sinus_rhythm",
            "atrial_fibrillation",
            "atrial_flutter",
            "sinus_bradycardia",
            "sinus_tachycardia",
            "first_degree_av_block",
            "left_bundle_branch_block",
            "right_bundle_branch_block",
            "premature_ventricular_contraction",
            "premature_atrial_contraction",
        ],
    },
}


def get_schema(name: str) -> dict[str, list[str] | str]:
    if name not in LABEL_SCHEMAS:
        raise ValueError(f"Unknown label schema: '{name}'. Available: {list(LABEL_SCHEMAS.keys())}")
    return LABEL_SCHEMAS[name]


def get_valid_labels(name: str) -> list[str]:
    schema = get_schema(name)
    labels = schema["labels"]
    if isinstance(labels, list):
        return labels
    raise ValueError(f"Schema '{name}' has invalid labels format")


def validate_label(label: str, schema_name: str) -> str:
    valid = get_valid_labels(schema_name)
    if label not in valid:
        raise ValueError(f"Invalid label '{label}' for schema '{schema_name}'. Valid labels: {valid}")
    return label


def list_schemas() -> list[str]:
    return list(LABEL_SCHEMAS.keys())
