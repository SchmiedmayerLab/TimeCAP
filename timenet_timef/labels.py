from __future__ import annotations

LABEL_SCHEMAS: dict[str, dict[str, list[str] | str]] = {
    "binary": {
        "description": "Binary yes/no classification",
        "labels": ["yes", "no"],
    },
}


def get_schema(name: str) -> dict[str, list[str] | str]:
    if name not in LABEL_SCHEMAS:
        raise ValueError(f"Unknown label schema: {name}")
    return LABEL_SCHEMAS[name]


def get_valid_labels(name: str) -> list[str]:
    schema = get_schema(name)
    labels = schema["labels"]
    if not isinstance(labels, list):
        raise ValueError(f"Invalid labels payload for schema {name}")
    return labels
