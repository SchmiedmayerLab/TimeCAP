# SensorTSLM Core Framework

Dataset-agnostic captioning pipeline for sensor time-series data.

## Setup

Install dependencies and set the dataset path before running:

```bash
python3 -m pip install -r requirements.txt
```

```bash
export MHC_DATASET_DIR="../hf-daily_max-nonwear=50"
```

## Usage

```bash
python3 captionizer.py
```

## Explorer

Use the interactive explorer to inspect one row at a time, switch signals, and see which detector events fired where on the time series.

Start it with:

```bash
python3 explorer.py --min-wear-pct=50.0
```

Useful controls:

- Use the bottom row slider or `<` / `>` buttons to move between dataset rows.
- Click a signal in the right-hand signal list or in the channel overview heatmap to switch channels.
- Use the Matplotlib zoom and pan tools on the main plot to inspect parts of the signal in detail.
- Click `reset` or press `home` to reset the zoom.
- Use the overlay buttons to toggle `trend`, `spike`, and `watch` overlays.
- Use the `stats`, `events`, `captions`, and `help` tabs in the details panel to switch what metadata is shown.
- Scroll inside the details panel with the mouse wheel or the `^` / `v` buttons.
