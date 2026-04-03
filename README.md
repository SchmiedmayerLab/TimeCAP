# SensorTSLM Core Framework

Dataset-agnostic captioning pipeline for sensor time-series data.

## Setup

Set the dataset path before running:

```bash
export MHC_DATASET_DIR="../hf-daily_max-nonwear=50"
```

## Usage

```bash
python captionizer.py
```

## Export To TimeF

```python
from pathlib import Path

from captionizer import Captionizer
from exporters.timef_export import TimeFExportConfig, export_caption_result

result, _ = captionizer.run(max_rows=5)
root = export_caption_result(
    result,
    TimeFExportConfig(
        output_root=Path("exports"),
        dataset_id="mhc_caption_runs",
    ),
)
print(root)
```
