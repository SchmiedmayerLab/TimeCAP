<!--
SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# TimeCAP Datasets

The caption datasets produced by this pipeline are hosted separately from this code repo. **MHCap**, the caption layer over the [OpenMHC](https://arxiv.org/abs/2607.16235) wearable dataset, will be hosted on **Harvard Dataverse** under the same data-use agreement as OpenMHC. It will be available to all qualified researchers free of charge once the OpenMHC data release is live.

| Dataset | Contents | Size | Use case | DOI |
|---|---|---|---|---|
| MHCap weekly | 7,005,763 captions over 144,412 weekly records (908,944 person-days, 2,081 participants) | To follow | Caption-augmented training and benchmarking of time-series language models | Available with the OpenMHC release |
| MHCap daily | Captions over daily minute-level records | To follow | The same, at daily resolution | Available with the OpenMHC release |
| MHCap tiny | Weekly and daily captions for 593 participants (7,134 weekly and 82,406 daily records) | ~256 MB weekly, ~350 MB daily | Quickstarts, smoke tests and inspecting the pipeline | Available with the OpenMHC release |
| WESADCap | 1,461 captions over 15 WESAD subjects | ~7 MB | Running the pipeline on lab physiology | To follow |

## How MHCap relates to OpenMHC

MHCap adds captions to OpenMHC. The full weekly and daily layers do not contain a second copy of the OpenMHC signals.

- **Signals.** Get the sensor data from OpenMHC itself. See its [DATASET.md](https://github.com/AshleyLab/OpenMHC/blob/main/DATASET.md).
- **Keys.** Every record carries `user_id` and `date`, which link it to the matching OpenMHC records.
- **Participants.** MHCap only covers participants who consented to broad data sharing, the same participants OpenMHC releases.
- **Weekly windows.** Weekly records cover 7-day windows, moved 7 days at a time, each with at least 5 valid days. OpenMHC's weekly window index (`window_index_w7_s7_d5.parquet`) uses the same rule.

The tiny set also includes the signal values, so it can be inspected on its own.

## Record schema

Each record is one recording: one participant-day in the daily layers, one participant-week in the weekly layers.

| Field | Type | Description |
|---|---|---|
| `row_id` | string | Record identifier, e.g. `mhc_weekly:<user_id>:<week_start>` |
| `user_id` | string | OpenMHC participant ID |
| `date` | string | The day of a daily record, or the first day of a weekly window |
| `values` | list of lists | Signal values, one list per channel (tiny set only) |
| `channel_names`, `display_names`, `units` | list of strings | Channel metadata, one entry per channel |
| `has_any_data` | list of booleans | Whether each channel has any data in the record |
| `channel_variance` | list of floats | Variance of each channel within the record |
| `annotations` | list of records | The captions, see below |
| `wear_pct`, `total_nonwear_minutes` | float | Daily layers only: wear time of the day |

Each entry in `annotations` has these fields:

| Field | Description |
|---|---|
| `caption_type` | `statistical`, `structural`, `semantic` or `cross_channel` |
| `text` | The caption |
| `channel_idxs` | Indices of the channels the caption describes |
| `label` | Machine-readable event label, e.g. `sleep_asleep` or `walking` |
| `window_start`, `window_end` | Start and end of the described interval on the record's time axis: hours for weekly records, minutes for daily records |

## Splits

- The tiny set ships `sharable_users_seed42_2026_tiny.json`, with 356 training, 59 validation and 178 test participants. The weekly tiny set also ships `weekly_labels_lookup_stride7.parquet`, the weekly label lookup used for the prediction tasks.
- For the full layers, use OpenMHC's canonical split file, `sharable_users_seed42_2026.json`. When exporting captions yourself, pass it with `--split_file`. The exporter then drops all other participants at load time.

## Loading

The record files are Parquet, with an Arrow copy for `datasets.load_from_disk`:

```python
from datasets import load_dataset

weekly = load_dataset("parquet", data_files="<weekly-dir>/*.parquet", split="train")
```

To browse records with the detector outputs drawn on the signals, point the explorer at a downloaded tiny folder:

```bash
MHC_WEEKLY_DATASET_DIR=<weekly-tiny-dir> python explorer.py --weekly
MHC_DATASET_DIR=<daily-tiny-dir> python explorer.py
```

## Regenerating the captions

Captions are generated deterministically, so the same code version and source data give identical output. To build them from your own copy of the source data, follow the export instructions in the [README](README.md).

## Data use agreement

MHCap is derived from OpenMHC and is shared under the same data-use agreement. Downloading it from Dataverse requires accepting that agreement. The captions are research artifacts and must not be used for clinical diagnosis or decisions about individuals.

## Citation

If you use MHCap, please cite this repository (see [CITATION.cff](CITATION.cff)) and the OpenMHC paper ([arXiv:2607.16235](https://arxiv.org/abs/2607.16235)).
