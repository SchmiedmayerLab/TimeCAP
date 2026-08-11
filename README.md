<!--

This source file is part of the TimeCAP open-source project

SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# TimeCAP

[![Build and Test](https://github.com/SchmiedmayerLab/TimeCAP/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/SchmiedmayerLab/TimeCAP/actions/workflows/static-analysis.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/SchmiedmayerLab/TimeCAP/blob/main/LICENSE.md)

Dataset-agnostic pipeline that turns raw multi-channel sensor recordings into
language-model-ready captions. The current reference implementation targets the
MyHeartCounts (MHC) wearable dataset at daily and weekly resolution.

## Pipeline overview

```
HuggingFace dataset ──► Transformer ──► Recording ──► Annotator ──► CaptionResult
                                                       │
                                                       ├── StructuralExtractor   (trends, spikes)
                                                       ├── SemanticExtractor     (active windows)
                                                       └── CrossChannelExtractor (workout / sleep)
```

- **`Transformer`** maps a source-dataset row to the internal `Recording` schema
  (per-channel float arrays + metadata).
- **`Annotator`** runs a list of `CaptionExtractor`s and returns one
  `Annotation` per (channel, time-window, caption-type) it observes.
- **Caption phrasing** is template-driven (`templates/templates.json`,
  `templates/templates_hourly.json`); each annotation deterministically picks a
  template variant from a row-derived seed.

Adding a new source dataset means writing one `Dataset`, one `Transformer`, and
a `ChannelConfig` describing channel names, units, and aggregators — no changes
to the core pipeline.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Point the loader at a HuggingFace MHC export:

```bash
export MHC_DATASET_DIR=<path-to-daily-hf-dataset>     # for daily
export MHC_WEEKLY_DATASET_DIR=<path-to-weekly-hf>     # for weekly
```

## Generating captions

Single-process export:

```bash
python scripts/export_captions.py \
    --variant weekly \
    --out exports/lean_full
```

Common flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--variant {daily,weekly}` | `weekly` | Which MHC resolution to caption. |
| `--out <dir>` | `exports/lean_full` | Output directory for Arrow shards. |
| `--max_rows <n>` | unset | Cap row count for a smoke test. |
| `--start <i>` / `--end <j>` | full range | Slice for parallel/sharded runs. |
| `--min_wear_pct <p>` | `0.0` | (daily) drop low-wear days. |
| `--min_valid_hours <h>` | `0` | (weekly) drop weeks with too few valid hours. |
| `--min_active_channels <k>` | `0` | Drop rows with fewer active channels. |
| `--split_file <json>` | unset | Use canonical sharable-user splits. |

Each shard writes an Arrow file under `<out>/recordings_*.arrow` containing the
`Recording` data plus all `Annotation`s.

### Parallel sharded export

For large jobs the helper script splits the dataset into `N` Slurm jobs:

```bash
export MHC_WEEKLY_DATASET_DIR=<path>
./scripts/export_captions_sharded.sh weekly 4 exports/lean_full
```

## Library usage

```python
from annotator import Annotator
from captionizer import Captionizer
from extractors.semantic import SemanticExtractor
from extractors.structural import StructuralExtractor
from mhc.constants import MHC_CHANNEL_CONFIG
from mhc.cross_channel import default_extractor
from mhc.dataset import MHCDataset
from mhc.transformer import MHCTransformer

dataset = MHCDataset()
annotator = Annotator([
    StructuralExtractor(MHC_CHANNEL_CONFIG),
    SemanticExtractor(MHC_CHANNEL_CONFIG),
    default_extractor(MHC_CHANNEL_CONFIG),
])
captionizer = Captionizer(dataset, MHCTransformer(), annotator)
result, _ = captionizer.run(max_rows=10)
```

## Inspecting the output

The interactive explorer steps through one row at a time, switches signals, and
overlays detector events on the time series:

```bash
python explorer.py --min-wear-pct=50.0
```

## WESAD

WESAD support adds a second dataset path for the same captioning pipeline. It
lets you work with the raw WESAD sensor recordings, preprocess them into the
stored `timef` format on demand, and then inspect or export them with the same
annotator/explorer flow used for MHC.

Typical workflow:

```bash
python3 explorer.py --wesad
python3 scripts/export_wesad_captions.py --out exports/wesad_poc
```

If you already have a stored dataset, you can still point the tools at it with
`--wesad-dataset-dir /path/to/wesad_timef`. If not, the WESAD dataset adapter
will fall back to raw subject folders under `data/WESAD` and the transformer
will run the preprocessing steps inline.

The WESAD mode includes:

- a dedicated `wesad/` dataset adapter and transformer
- WESAD-specific templates and synthesizers for stress, recovery, and amusement
- support for either a prebuilt stored dataset or raw WESAD folders
- the same downstream explorer/export pipeline used for the other datasets

## Layout

```
captionizer.py        Orchestration (Dataset + Transformer + Annotator)
annotator.py          Runs a list of extractors over one Recording
extractors/           Caption extractors (statistical / structural / semantic / cross-channel)
detectors/            Trend and spike detectors used by structural extractor
synthesizers/         Cross-channel caption synthesizers (workout, sleep, …)
templates/            Caption phrasing templates
mhc/                  MHC daily dataset adapter
mhc_weekly/           MHC weekly dataset adapter
timef/                Internal Recording / CaptionResult schema
exporters/            Arrow shard writer
scripts/              CLI entry points (caption export, training, eval)
```

## Contributing

Contributions to this project are welcome. Please make sure to read the [contribution guidelines](https://github.com/SchmiedmayerLab/.github/blob/main/CONTRIBUTING.md) and the [contributor covenant code of conduct](https://github.com/SchmiedmayerLab/.github/blob/main/CODE_OF_CONDUCT.md) first. You can find a list of contributors in the [CONTRIBUTORS.md](CONTRIBUTORS.md) file.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for more information.

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff), which GitHub surfaces through the [*Cite this repository*](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) button.

## Our Research

For more information, visit the [Schmiedmayer Lab GitHub organization](https://github.com/SchmiedmayerLab).

![Schmiedmayer Lab](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/footer-light.png#gh-light-mode-only)
![Schmiedmayer Lab](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/footer-dark.png#gh-dark-mode-only)
