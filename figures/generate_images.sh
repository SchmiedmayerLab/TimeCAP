# -----------------
# MHC figures
# -----------------

python scripts/render_caption_visualization.py --row-index 3402 --example auto --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/sleep-figure1.png
python scripts/render_caption_visualization.py --row-index 3662 --example auto --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/sleep-figure2.png
python scripts/render_caption_visualization.py --row-index 3724 --example auto --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/sleep-figure3.png

python scripts/render_caption_visualization.py --row-index 3463 --example walking --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/walking1.png
python scripts/render_caption_visualization.py --row-index 3402 --example walking --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/walking2.png

python scripts/render_caption_visualization.py --row-index 3402 --example auto --channels WHR WSteps WDist WEnergy Cycle --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/cycling.png
python scripts/render_caption_visualization.py --row-index 8229 --example auto --channels WHR WSteps WDist WEnergy Run --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/running.png
python scripts/render_caption_visualization.py --row-index 2927 --example auto --channels WHR WSteps WDist WEnergy Elliptical --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/elliptical.png
python scripts/render_caption_visualization.py --row-index 8574 --example auto --channels WHR WSteps WDist WEnergy HIIT --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/hiit.png
python scripts/render_caption_visualization.py --row-index 8844 --example auto --channels WHR WSteps WDist WEnergy Yoga --include-statistical --caption-types statistical structural semantic cross_channel --max-cards 8 --out /Users/deckk/SensorTSLM/figures/MHC/yoga.png

# -----------------
# WESAD figures
# -----------------

python scripts/render_caption_visualization.py --dataset wesad --wesad-arrow-dir /Users/deckk/SensorTSLM/exports/wesad_full --row-index 0 --example auto --out /Users/deckk/SensorTSLM/figures/WESAD/wesad_auto.png
