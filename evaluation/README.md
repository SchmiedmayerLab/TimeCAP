# Evaluation Pipeline

Prompt-based downstream evaluation for plotted wearable sensor data.

The current runner evaluates models by rendering sensor channels as plots, asking a structured question, parsing the text answer, and computing classification metrics. This is the repo's **benchmark pipeline**. It is separate from the caption-generation pipeline and separate from any later OpenTSLM training/fine-tuning.

## What This Pipeline Does

Implemented today:

1. **Metabolic evaluation** on weekly hourly MHC summaries with static survey labels.

Not implemented yet:

1. Table 4 sleep 2-class task.
2. Table 5 retrieval.
3. Official Stage 1 / Stage 2 trained rows.

## Directory Structure

```text
evaluation/
├── run_eval.py
├── evaluator.py
├── README.md
├── tasks/
│   ├── base.py
│   └── metabolic.py
├── metrics/
│   └── classification.py
└── io/
    └── writer.py
```

Dataset adapters used by the runner:

```text
time_series_datasets/
├── mhc_metabolic_dataset.py
└── mhc_caption_qa_dataset.py
```

Only `mhc_metabolic_dataset.py` is used by `run_eval.py`.

## Currently Registered Datasets And Tasks

Datasets in `run_eval.py`:

- `mhc_metabolic_biological_sex`
- `mhc_metabolic_bmi`
- `mhc_metabolic_diabetes`
- `mhc_metabolic_hypertension`
- `mhc_metabolic_cardiovascular`

Tasks in `run_eval.py`:

- `metabolic_biological_sex`
- `metabolic_bmi`
- `metabolic_diabetes`
- `metabolic_hypertension`
- `metabolic_cardiovascular`

## Inputs And Ground Truth

### Metabolic

- Dataset class: `MHCMetabolicDataset`
- Input: one week of hourly MHC data
- Channels seen by model: **all 19 weekly channels**
  - 7 continuous channels
  - 2 sleep binary channels
  - 10 workout binary channels
- Ground truth: static survey labels from `data/labels_dev_subset.parquet`
- Output format: one label option on the last line

This currently supports:

- `BiologicalSex`
- `Diabetes`
- `Hypertension`
- `cardiovascular_disease`
- `BMI_categories`

Still missing for the paper's Table 3 scope:

- `happiness_categories`
- `blood_pressure_categories`
- `sleep_time_categories`

AFib exists in the labels parquet, but the class balance is currently too extreme to be useful in this dev subset.

## Usage

All runs go through `evaluation/run_eval.py`.

### Metabolic

```bash
python evaluation/run_eval.py \
    --dataset mhc_metabolic_biological_sex \
    --task metabolic_biological_sex \
    --model gpt-4o \
    --split test
```

Run all metabolic labels in one go:

```bash
for PAIR in \
    "mhc_metabolic_biological_sex metabolic_biological_sex" \
    "mhc_metabolic_diabetes metabolic_diabetes" \
    "mhc_metabolic_hypertension metabolic_hypertension" \
    "mhc_metabolic_cardiovascular metabolic_cardiovascular" \
    "mhc_metabolic_bmi metabolic_bmi"; do
  DS=$(echo $PAIR | cut -d' ' -f1)
  TASK=$(echo $PAIR | cut -d' ' -f2)
  python evaluation/run_eval.py --dataset $DS --task $TASK --model gpt-4o --split test
done
```

Or individually:

```bash
python evaluation/run_eval.py --dataset mhc_metabolic_biological_sex  --task metabolic_biological_sex  --model gpt-4o
python evaluation/run_eval.py --dataset mhc_metabolic_diabetes        --task metabolic_diabetes        --model gpt-4o
python evaluation/run_eval.py --dataset mhc_metabolic_hypertension    --task metabolic_hypertension    --model gpt-4o
python evaluation/run_eval.py --dataset mhc_metabolic_cardiovascular  --task metabolic_cardiovascular  --model gpt-4o
python evaluation/run_eval.py --dataset mhc_metabolic_bmi             --task metabolic_bmi             --model gpt-4o
```

Quick development run:

```bash
python evaluation/run_eval.py \
    --dataset mhc_metabolic_biological_sex \
    --task metabolic_biological_sex \
    --model gpt-4o \
    --max-samples 50 \
    --max-users 50
```

## CLI Reference

| Flag | Description |
|---|---|
| `--dataset` | registered dataset name |
| `--task` | registered task name |
| `--model` | model id |
| `--base-url` | OpenAI-compatible endpoint for client models |
| `--split` | `train`, `validation`, `test` |
| `--max-samples` | cap evaluated samples |
| `--max-users` | scan cap for metabolic dataset construction |
| `--dataset-dir` | override dataset path |
| `--labels-path` | override weekly labels parquet |
| `--device` | device for local HF models |
| `--results-dir` | output root |

## Output

Results are written to:

```text
results/<task>/<model>/<timestamp>/
```

Each run writes:

- `metrics.json`
- `predictions.json`

## What This Pipeline Does Not Do

The runner currently calls:

```python
model.process(prompt=prompt, signals=signals)
```

By default it does **not** pass captions into the model (prompt-only baseline).

To run with captions, first generate them:

```bash
# Generate placeholder captions (dummy text — replace with trained captionizer output)
python evaluation/generate_captions_placeholder.py

# Then run eval with captions prepended to the prompt
python evaluation/run_eval.py \
    --dataset mhc_metabolic_biological_sex \
    --task metabolic_biological_sex \
    --model gpt-4o \
    --captions-dir data/mhc_dev_subset_captions
```

**TODO (once Stage 1 training is complete):** replace `evaluation/generate_captions_placeholder.py`
with a script that runs the trained captionizer over the weekly dataset and writes real captions
to `data/mhc_dev_subset_captions/captions.parquet`. The eval pipeline requires no other changes.
