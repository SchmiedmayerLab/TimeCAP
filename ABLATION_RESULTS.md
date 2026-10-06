# Caption-family ablation — 40/40 cells complete

Parse rate across all cells: **0.9904 – 1.0000**

| task | metric | structural | semantic | crosschannel | bs2 | trivial | spread | verdict |
|---|---|---|---|---|---|---|---|---|
| afib | f1 | 0.0711 ± 0.0031 | 0.0541 ± 0.0029 | 0.0745 ± 0.0031 | 0.0689 ± 0.0032 | 0.0579 | 0.0203 | separated |
| bmi | qwk | 0.5809 ± 0.0037 | 0.5973 ± 0.0035 | 0.5964 ± 0.0035 | 0.6015 ± 0.0034 | 0.0000 | 0.0206 | separated |
| biological_sex | f1 | 0.8708 ± 0.0014 | 0.8672 ± 0.0014 | 0.8677 ± 0.0014 | 0.8694 ± 0.0014 | 0.8695 | 0.0036 | **within noise** |
| cad | f1 | 0.2990 ± 0.0050 | 0.3685 ± 0.0069 | 0.2944 ± 0.0048 | 0.3221 ± 0.0058 | 0.1979 | 0.0741 | separated |
| cardiovascular | f1 | 0.4637 ± 0.0045 | 0.3663 ± 0.0057 | 0.4438 ± 0.0046 | 0.4108 ± 0.0050 | 0.3747 | 0.0975 | separated |
| diabetes | f1 | 0.2752 ± 0.0052 | 0.2715 ± 0.0057 | 0.2540 ± 0.0052 | 0.2715 ± 0.0055 | 0.2054 | 0.0211 | separated |
| happiness | qwk | 0.0897 ± 0.0050 | 0.0435 ± 0.0053 | 0.0243 ± 0.0055 | 0.0761 ± 0.0053 | 0.0000 | 0.0653 | separated |
| hypertension | f1 | 0.5458 ± 0.0041 | 0.5241 ± 0.0044 | 0.5428 ± 0.0041 | 0.5617 ± 0.0040 | 0.5878 | 0.0376 | separated |
| sleep_diagnosis | f1 | 0.2740 ± 0.0051 | 0.2424 ± 0.0051 | 0.2679 ± 0.0051 | 0.2635 ± 0.0051 | 0.3437 | 0.0316 | separated |
| work | f1 | 0.8409 ± 0.0016 | 0.8374 ± 0.0016 | 0.8378 ± 0.0017 | 0.8413 ± 0.0016 | 0.8343 | 0.0039 | **within noise** |

The `trivial` column is the headline metric's own constant-answer score. It is context, not a skill test — F1 ignores true negatives, so clearing it does not mean a model learned anything. The next table is the actual test.

## Does anything beat a constant answer?

Best checkpoint per task on a metric a constant answer cannot game: accuracy vs. the majority-class rate (binary), QWK vs. 0 (ordinal). Multiclass tasks are omitted — metrics.json lacks the class counts needed to state a majority rate.

| task | metric | best | trivial | delta | significance | beats trivial? |
|---|---|---|---|---|---|---|
| afib | accuracy | 0.6845 (semantic) | 0.9702 | -0.2857 | -114.6σ | **no** |
| bmi | qwk | 0.6015 (bs2) | 0.0000 | +0.6015 | +178.3σ | yes |
| biological_sex | accuracy | 0.7891 (structural) | 0.7691 | +0.0200 | +9.6σ | yes |
| cad | accuracy | 0.8455 (semantic) | 0.8902 | -0.0447 | -23.3σ | **no** |
| cardiovascular | accuracy | 0.7690 (semantic) | 0.7694 | -0.0004 | -0.2σ | **no** |
| diabetes | accuracy | 0.7049 (semantic) | 0.8863 | -0.1814 | -63.2σ | **no** |
| happiness | qwk | 0.0897 (structural) | 0.0000 | +0.0897 | +17.9σ | yes |
| hypertension | accuracy | 0.6655 (bs2) | 0.5837 | +0.0818 | +28.8σ | yes |
| sleep_diagnosis | accuracy | 0.7513 (bs2) | 0.7925 | -0.0412 | -19.1σ | **no** |
| work | accuracy | 0.7578 (bs2) | 0.7157 | +0.0421 | +19.4σ | yes |

**No skill over trivial:** `afib`, `cad`, `cardiovascular`, `diabetes`, `sleep_diagnosis` — caption-family differences on these tasks are differences between models that do not beat a constant answer.

## Tasks where the four checkpoints are within noise

`biological_sex`, `work`

Spread is smaller than 2x the standard error of a pairwise difference, so the caption families are not distinguishable here.

## Sample counts

| task | n_evaluated | parse_rate range |
|---|---|---|
| afib | 34560 | 1.0000 – 1.0000 |
| bmi | 37510 | 0.9993 – 1.0000 |
| biological_sex | 41162 | 1.0000 – 1.0000 |
| cad | 34560 | 1.0000 – 1.0000 |
| cardiovascular | 34560 | 1.0000 – 1.0000 |
| diabetes | 25711 | 0.9904 – 1.0000 |
| happiness | 28610 | 1.0000 – 1.0000 |
| hypertension | 25711 | 0.9999 – 1.0000 |
| sleep_diagnosis | 37776 | 1.0000 – 1.0000 |
| work | 37790 | 1.0000 – 1.0000 |