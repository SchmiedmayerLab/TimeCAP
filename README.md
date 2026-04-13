# SensorTSLM

Repository for two related but distinct workflows on wearable sensor data:

1. A captioning pipeline that turns MHC sensor recordings into structured, algorithmic captions.
2. A downstream evaluation pipeline that asks VLMs classification questions over plotted sensor signals.

This repo currently contains the data plumbing, visualization utilities, and prompt-based evaluation code. It does **not** yet contain the full OpenTSLM training / fine-tuning pipeline needed to reproduce the paper's official Stage 1 / Stage 2 rows.

## Current Scope

| Area | Status | Notes |
|---|---|---|
| Daily MHC loading + transform | implemented | 19-channel daily recordings from `data/mhc_dev_subset_hf` |
| Caption extractors | implemented | statistical, structural, semantic |
| Caption visualization | implemented | `visualizer.py`, `explorer.py` |
| Caption export for training | implemented | `exporters/lean.py` writes Arrow shards |
| OpenTSLM caption dataset adapter | implemented | `time_series_datasets/mhc_caption_qa_dataset.py` |
| Prompt-based activity eval | implemented | Table 2 style yes/no activity windows |
| Prompt-based metabolic eval | partially implemented | Table 3 subset wired; some label variants still missing |
| OpenTSLM Stage 1 training | missing in this repo | required for official caption-trained rows |
| Stage 2 task fine-tuning | missing in this repo | required for official fine-tuned rows |
| Retrieval evaluation | missing | no log-prob / caption scoring backend yet |

## Repo Map

| Path | Purpose |
|---|---|
| `captionizer.py` | Orchestrates caption generation over transformed recordings |
| `annotator.py` | Runs the extractor stack |
| `extractors/` | Statistical, structural, semantic caption generation |
| `detectors/` | Spike / trend primitives used by extractors |
| `visualizer.py` | Static qualitative plots with attached annotations |
| `explorer.py` | Interactive inspection tool for one daily sample at a time |
| `evaluation/run_eval.py` | Unified entry point for downstream benchmark runs |
| `evaluation/tasks/` | Prompt + parsing + metrics per task |
| `models/` | OpenAI-compatible client model and local HF VLM wrapper |
| `time_series_datasets/` | Dataset adapters for caption training and evaluation |
| `notebooks/01_downstream_label_exploration.ipynb` | Current project/eval status notebook with figures and LaTeX mapping |

## Data

Expected local data paths:

- `data/mhc_dev_subset_hf`: daily MHC subset
- `data/mhc_dev_subset_weekly_hf`: weekly MHC subset
- `data/labels_dev_subset.parquet`: weekly metabolic labels

The current dev subset sizes in this workspace are:

- daily rows: `130,102`
- weekly rows: `29,009`
- labelled users: `352`

## Setup

Install dependencies into your environment:

```bash
python3 -m pip install -r requirements.txt
```

If you use the checked-in virtualenv in this workspace:

```bash
./sensorlm-env/bin/python -m pip install -r requirements.txt
```

For API-backed evaluation models, set `OPENAI_API_KEY` in `.env`.

## Main Workflows

### 1. Explore and visualize daily captioning

Interactive explorer:

```bash
python3 explorer.py --min-wear-pct=50.0
```

The explorer lets you:

- move through daily recordings
- inspect channels and overlays
- see generated captions per signal

### 2. Run the downstream benchmark

Activity evaluation:

```bash
python3 evaluation/run_eval.py \
    --dataset mhc_activity \
    --task activity \
    --model gpt-4o \
    --split test
```

Metabolic evaluation:

```bash
python3 evaluation/run_eval.py \
    --dataset mhc_metabolic_biological_sex \
    --task metabolic_biological_sex \
    --model gpt-4o \
    --split test
```

More metabolic variants currently wired:

- `mhc_metabolic_diabetes` / `metabolic_diabetes`
- `mhc_metabolic_hypertension` / `metabolic_hypertension`
- `mhc_metabolic_cardiovascular` / `metabolic_cardiovascular`
- `mhc_metabolic_bmi` / `metabolic_bmi`

Evaluation results are written under `results/<task>/<model>/<timestamp>/`.

### 3. Inspect project status and paper-table coverage

Open:

- `notebooks/01_downstream_label_exploration.ipynb`
- `evaluation/README.md`

The notebook is the best current overview of:

- what is implemented vs missing
- which parts map to which LaTeX table
- which qualitative figures can be reused in the paper
- the distinction between general project gaps and eval-pipeline-specific gaps

## What Is Not Yet Implemented

These are the main blockers for full paper reproduction:

- OpenTSLM Stage 1 caption training
- Stage 2 task fine-tuning
- exact Table 2 per-activity reporting
- Table 3 extensions for `happiness_categories`, `blood_pressure_categories`, `sleep_time_categories`
- Table 4 sleep 2-class and activity 8-class tasks
- Table 5 retrieval scoring

## Notes On `mhc_metabolic_dataset.py`

`time_series_datasets/mhc_metabolic_dataset.py` is actively used by `evaluation/run_eval.py` for the weekly metabolic benchmark. It is needed for the Table 3-style weekly tasks. It is not used by the daily captioning pipeline.

## Notes On `annotations` In Evaluation

The current evaluation runner does **not** pass generated annotations into the model. That is intentional: the current benchmark is a raw-signal, prompt-only VLM baseline.

Annotations become relevant only if you later want:

- caption-assisted prompts
- retrieval against candidate captions
- OpenTSLM-style caption-trained models
