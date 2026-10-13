<!--
SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# TimeCAP Datasets

The caption datasets produced by this pipeline are hosted separately from this code repo. **MHCap**, the caption layer over the [OpenMHC](https://arxiv.org/abs/2607.16235) wearable dataset, will be hosted on **Harvard Dataverse** under the same data-use agreement as OpenMHC. It will be available to all qualified researchers free of charge once the OpenMHC data release is live.

| Dataset | Contents | Size | Use case | DOI |
|---|---|---|---|---|
| MHCap weekly | 7,005,763 captions over 144,412 weekly records (908,944 person-days, 2,081 participants) | To follow | Caption-augmented training and benchmarking of time-series language models | Available with the OpenMHC release |
| MHCap tiny | Weekly captions for 593 participants (7,134 weekly records) | To follow | Quickstarts, smoke tests and inspecting the caption format | Available with the OpenMHC release |
| WESADCap | 1,461 captions over 15 WESAD subjects | ~7 MB | Running the pipeline on lab physiology | To follow |

## How MHCap relates to OpenMHC

MHCap adds captions to OpenMHC. It contains no sensor signals, including in the tiny set.

- **Signals.** Get the sensor data from OpenMHC itself. See its [DATASET.md](https://github.com/AshleyLab/OpenMHC/blob/main/DATASET.md).
- **Keys.** Every record carries `user_id` and `date`. Dates follow OpenMHC's per-participant date shift, so they match the dates in the OpenMHC release.
- **Participants.** MHCap only covers participants who consented to broad data sharing, the same participants OpenMHC releases.
- **Weekly windows.** Weekly records cover 7-day windows with at least 5 valid days. They were built independently of OpenMHC's weekly window index (`window_index_w7_s7_d5.parquet`) and do not match it week for week, so MHCap ships its own window index for joining captions to OpenMHC signals.

## Daily records

The experiments in the paper use weekly records only, so the first release contains weekly captions. The pipeline can also caption daily minute-level records (`--variant daily`, see the [README](README.md)), and a daily layer can be generated from OpenMHC in the same way.

## Record schema

Each record is one participant-week. The exact file layout will be documented with the release.

| Field | Type | Description |
|---|---|---|
| `user_id` | string | OpenMHC participant ID |
| `date` | string | First day of the weekly window, date-shifted as in OpenMHC |
| `channel_names`, `display_names`, `units` | list of strings | Channel metadata, one entry per channel |
| `annotations` | list of records | The captions, see below |

Each entry in `annotations` has these fields:

| Field | Description |
|---|---|
| `caption_type` | `statistical`, `structural`, `semantic` or `cross_channel` |
| `text` | The caption |
| `channel_idxs` | Indices of the channels the caption describes |
| `label` | Machine-readable event label, e.g. `sleep_asleep` or `walking` |
| `window_start`, `window_end` | Start and end of the described interval, in hours from the start of the week |

Caption text contains no calendar dates, only hours within the week.

## Splits

- The tiny set ships `sharable_users_seed42_2026_tiny.json`, with 356 training, 59 validation and 178 test participants.
- For the full layer, use OpenMHC's canonical split file, `sharable_users_seed42_2026.json`. When exporting captions yourself, pass it with `--split_file`. The exporter then drops all other participants at load time.

## Loading

The caption files are Parquet:

```python
from datasets import load_dataset

weekly = load_dataset("parquet", data_files="<weekly-dir>/*.parquet", split="train")
```

To browse captions drawn on the signals, export records from your own copy of the OpenMHC data (see the [README](README.md)) and open them in the explorer:

```bash
MHC_WEEKLY_DATASET_DIR=<weekly-export-dir> python explorer.py --weekly
```

## Regenerating the captions

Captions are generated deterministically, so the same code version and source data give identical output. Each caption picks its template wording from a seed derived from the participant ID and the record date. Regenerating captions from the date-shifted OpenMHC release therefore yields the same events and values, but some captions can use a different template wording than the released files. To build captions from your own copy of the source data, follow the export instructions in the [README](README.md).

## Data use agreement

MHCap is derived from OpenMHC and is shared under the same data-use agreement. Downloading it from Dataverse requires accepting that agreement. The captions are research artifacts and must not be used for clinical diagnosis or decisions about individuals.

## Citation

If you use MHCap, please cite this repository (see [CITATION.cff](CITATION.cff)) and the OpenMHC paper ([arXiv:2607.16235](https://arxiv.org/abs/2607.16235)).
