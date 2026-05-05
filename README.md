# TimeCap

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
