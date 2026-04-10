# Evaluation Pipeline

LLM-based evaluation pipeline for the SensorTSLM captioning system. Models are
presented with windowed Apple Watch sensor time series and asked structured
questions; answers are scored against ground truth derived directly from
HealthKit labels — no manual annotation required.

## Overview

The pipeline evaluates two types of tasks:

1. **Activity recognition** (implemented): Binary yes/no classification over
   60-minute sensor windows. Ground truth from Apple Watch `workout:*` binary
   channels.

2. **Sleep detection** (stub — pending PI alignment): Binary classification of
   sleep vs. awake windows. Ground truth from `sleep:asleep` binary channel.

3. **Caption quality** (stub — pending PI alignment): Free-text caption
   generation evaluated against pipeline-generated reference captions from
   `exporters/lean.py`.

All tasks share the same runner, metrics infrastructure, and result writer.
Adding a new task requires only a dataset class in `time_series_datasets/` and
a task class in `evaluation/tasks/`.

## Directory Structure

```
evaluation/
├── run_eval.py                  # Unified entry point (dataset/model/task via CLI)
├── evaluator.py                 # Evaluator: inference loop + metric aggregation
├── config.py                    # EvalConfig, ModelConfig (frozen dataclasses)
├── README.md                    # This file
├── tasks/
│   ├── base.py                  # EvalTask ABC + EvalSample dataclass
│   ├── activity.py              # Prompt, yes/no parsing, F1 aggregation
│   ├── sleep.py                 # Stub — design questions pending
│   └── caption.py               # Stub — design questions pending
├── metrics/
│   ├── classification.py        # F1, Precision, Recall, Accuracy + bootstrap SE
│   └── text.py                  # Stub — ROUGE-L, BERTScore (pending)
└── io/
    └── writer.py                # ResultsWriter → results/<task>/<model>/<timestamp>/
```

The dataset classes live next to the existing `MHCCaptionQADataset`:

```
time_series_datasets/
├── mhc_caption_qa_dataset.py    # Existing — OpenTSLM training dataset
└── mhc_activity_dataset.py      # New — 60-min windows for activity evaluation
```

## Dependencies

No new packages beyond the existing `requirements.txt`, except `scikit-learn`
for classification metrics (already listed as a transitive dependency of
several existing packages). Install if missing:

```bash
pip install scikit-learn
```

## Usage

All evaluation runs go through the single entry point `run_eval.py`. Dataset,
model, and task are selected via CLI flags.

> **Note on scan time:** `MHCActivityDataset` builds its sample index by
> scanning the full MHC dataset on startup. With 130k recordings this can take
> several minutes. Use `--max-recordings` to limit the scan during development.
> For full runs, always set `--max-recordings` explicitly so results are
> reproducible. A pre-filtered HuggingFace subset based on `has_any_data` flags
> is planned to eliminate the scan entirely (TBD).

```bash
# Activity recognition with GPT-4o — full test split:
python evaluation/run_eval.py \
    --dataset mhc_activity \
    --model gpt-4o \
    --task activity

# Quick dev run — 50 samples, scan only 500 recordings:
python evaluation/run_eval.py \
    --dataset mhc_activity \
    --model gpt-4o \
    --task activity \
    --max-samples 50 \
    --max-recordings 500

# Local Llama via HuggingFace (GPU):
python evaluation/run_eval.py \
    --dataset mhc_activity \
    --model meta-llama/Llama-3.2-1B \
    --task activity \
    --device cuda

# Local Ollama endpoint:
python evaluation/run_eval.py \
    --dataset mhc_activity \
    --model llama3.2 \
    --task activity \
    --base-url http://localhost:11434/v1

# Validation split:
python evaluation/run_eval.py \
    --dataset mhc_activity \
    --model gpt-4o \
    --task activity \
    --split validation
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Dataset: `mhc_activity` |
| `--task` | required | Task: `activity` |
| `--model` | required | Model name: `gpt-4o`, `meta-llama/Llama-3.2-1B`, `llama3.2`, … |
| `--base-url` | `https://api.openai.com/v1` | API endpoint for client models |
| `--split` | `test` | Dataset split: `train`, `validation`, `test` |
| `--max-samples` | `None` | Cap on samples evaluated (None = all) |
| `--max-recordings` | `None` | Cap on MHC recordings scanned for the index |
| `--window-size` | `60` | Window duration in minutes |
| `--dataset-dir` | `data/mhc_dev_subset_hf` | Path to HuggingFace MHC dataset |
| `--device` | `cpu` | Device for local models: `cuda`, `mps`, `cpu` |
| `--results-dir` | `results/` | Root directory for output files |

## Ground Truth

| Task | Source | Type |
|------|--------|------|
| Activity | `workout:HKWorkoutActivity*` channels | Apple Watch binary label (0/1 per minute) |
| Sleep | `sleep:asleep` channel | Apple Watch binary label (0/1 per minute) |
| Caption | Pipeline captions via `exporters/lean.py` | Generated reference text |

Activity and sleep labels are recorded directly by Apple Watch and stored in
HealthKit — they require no manual annotation. Each 60-minute evaluation window
is labelled `"yes"` if a matching workout event overlaps the window, `"no"`
otherwise. Positive and negative samples are class-balanced per recording.

User-level train/val/test splits (80/10/10) use the same SHA-1 bucketing as
`MHCCaptionQADataset` to prevent leakage across the captioning and evaluation
pipelines.

## Metrics

### Classification tasks (activity, sleep)

| Metric | Description |
|--------|-------------|
| `f1` | Binary F1 score (primary metric — robust to class imbalance) |
| `precision` | Precision for the positive class |
| `recall` | Recall for the positive class |
| `accuracy` | Overall accuracy |
| `*_se` | Bootstrap standard error (1 000 resamples, seed 42) for each metric |
| `n_samples` | Total samples evaluated |
| `n_positive_gt` | Positive samples in ground truth |
| `n_positive_pred` | Positive predictions |
| `n_unknown` | Samples where model output could not be parsed |
| `parse_rate` | Fraction of samples with a parseable prediction |

F1 is the primary reported metric because evaluation windows are constructed
to be class-balanced, but individual workout types vary in base rate.
Bootstrap SEs follow the MHC-Benchmark convention and are reported alongside
point estimates in all result files.

## Output

Results are written under `results/<task>/<model>/<timestamp>/`:

```
results/
└── activity/
    └── gpt-4o/
        └── 20260409_143022/
            ├── metrics.json       # Aggregated metrics (F1, SE, …)
            └── predictions.json   # Per-sample ground truth + prediction
```

`metrics.json` is directly JSON-serialisable (flat dict of floats and ints).
`predictions.json` includes the raw model output alongside the parsed
prediction for manual error analysis.

## Adding a New Task

1. Create a dataset class in `time_series_datasets/` that yields `EvalSample`
   objects (see `mhc_activity_dataset.py` as reference).
2. Create a task class in `evaluation/tasks/` inheriting `EvalTask` and
   implementing `build_prompt`, `parse_prediction`, `aggregate_metrics`.
3. Register both in the `DATASET_REGISTRY` and `TASK_REGISTRY` dicts at the
   top of `evaluation/run_eval.py`.

The evaluator, writer, and metrics modules require no changes.
