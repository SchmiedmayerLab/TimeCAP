<!--

This source file is part of the TimeCAP open-source project

SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Model metrics (recomputed from prediction CSVs)

Per-week macro-F1 / accuracy for binary tasks (label-symmetric: a majority-class predictor scores ~accuracy/(1+accuracy) on the dominant class and 0 on the other, so macro-F1 is bounded by the imbalance).
Per-user AUROC / AUPRC use the per-user mean of binary predictions as the score and the user's static label as ground truth.
Binary unknowns count as a negative-class prediction (0) at both granularities — abstention is penalized, not dropped.


## Binary tasks

| Task | Model | n_users | parse_rate | macro-F1 | Acc | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| Atrial Fibrillation | OpenTSLM-Cap-3B | 464 | 100.0\% | 44.39 | 67.97 | 64.08 | 7.33 |
| Atrial Fibrillation | OpenTSLM-Cap-1B | 464 | 100.0\% | 43.68 | 67.76 | 68.36 | 7.24 |
| Atrial Fibrillation | OpenTSLM-1B | 464 | 100.0\% | 44.99 | 71.41 | 59.82 | 4.91 |
| Atrial Fibrillation | OpenTSLM-3B | 464 | 100.0\% | 41.79 | 62.20 | 61.72 | 5.40 |
| Biological Sex | OpenTSLM-Cap-3B | 587 | 100.0\% | 64.87 | 78.29 | 77.86 | 51.24 |
| Biological Sex | OpenTSLM-Cap-1B | 587 | 100.0\% | 64.59 | 78.71 | 79.24 | 51.72 |
| Biological Sex | OpenTSLM-1B | 587 | 100.0\% | 65.20 | 78.12 | 78.90 | 49.99 |
| Biological Sex | OpenTSLM-3B | 587 | 100.0\% | 61.24 | 78.40 | 75.47 | 49.80 |
| Coronary Artery Disease | OpenTSLM-Cap-3B | 464 | 100.0\% | 55.60 | 70.78 | 62.46 | 12.57 |
| Coronary Artery Disease | OpenTSLM-Cap-1B | 464 | 100.0\% | 58.83 | 76.02 | 65.88 | 15.76 |
| Coronary Artery Disease | OpenTSLM-1B | 464 | 99.9\% | 61.11 | 81.91 | 63.38 | 16.62 |
| Coronary Artery Disease | OpenTSLM-3B | 464 | 100.0\% | 52.96 | 66.82 | 59.94 | 9.69 |
| Diabetes | OpenTSLM-Cap-3B | 279 | 100.0\% | 59.12 | 74.07 | 59.44 | 12.06 |
| Diabetes | OpenTSLM-Cap-1B | 279 | 100.0\% | 53.14 | 67.60 | 64.56 | 12.60 |
| Diabetes | OpenTSLM-1B | 279 | 100.0\% | 51.35 | 65.91 | 57.19 | 10.56 |
| Diabetes | OpenTSLM-3B | 279 | 100.0\% | 53.02 | 67.89 | 59.56 | 11.39 |
| Hypertension | OpenTSLM-Cap-3B | 279 | 100.0\% | 65.52 | 68.38 | 70.48 | 54.89 |
| Hypertension | OpenTSLM-Cap-1B | 279 | 100.0\% | 64.56 | 66.54 | 69.89 | 48.76 |
| Hypertension | OpenTSLM-1B | 279 | 100.0\% | 64.01 | 65.67 | 67.96 | 48.01 |
| Hypertension | OpenTSLM-3B | 279 | 100.0\% | 64.34 | 66.46 | 69.09 | 54.86 |
| Cardiovascular Disease | OpenTSLM-Cap-3B | 464 | 100.0\% | 62.00 | 70.22 | 67.19 | 28.94 |
| Cardiovascular Disease | OpenTSLM-Cap-1B | 464 | 100.0\% | 61.65 | 72.62 | 65.15 | 29.99 |
| Cardiovascular Disease | OpenTSLM-1B | 464 | 99.7\% | 61.13 | 76.37 | 62.09 | 26.79 |
| Cardiovascular Disease | OpenTSLM-3B | 464 | 100.0\% | 58.62 | 67.13 | 65.28 | 26.25 |
| Sleep Diagnosis | OpenTSLM-Cap-3B | 530 | 100.0\% | 57.01 | 75.24 | 60.43 | 26.53 |
| Sleep Diagnosis | OpenTSLM-Cap-1B | 530 | 100.0\% | 55.68 | 75.11 | 59.52 | 25.35 |
| Sleep Diagnosis | OpenTSLM-1B | 530 | 100.0\% | 56.81 | 74.57 | 58.86 | 26.13 |
| Sleep Diagnosis | OpenTSLM-3B | 530 | 100.0\% | 56.98 | 74.68 | 57.95 | 24.99 |
| Employment | OpenTSLM-Cap-3B | 531 | 100.0\% | 64.97 | 74.40 | 67.99 | 42.36 |
| Employment | OpenTSLM-Cap-1B | 531 | 100.0\% | 66.50 | 75.79 | 66.30 | 41.52 |
| Employment | OpenTSLM-1B | 531 | 100.0\% | 64.17 | 74.55 | 64.74 | 38.49 |
| Employment | OpenTSLM-3B | 531 | 100.0\% | 66.84 | 75.44 | 64.28 | 36.84 |

## Ordinal tasks

| Task | Model | n_users | parse_rate | wk-rho | wk-QWK | user-rho | user-QWK | user-MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BMI (5 levels) | OpenTSLM-Cap-3B | 557 | 100.0\% | 61.59 | 62.40 | 63.94 | 62.30 | 0.513 |
| BMI (5 levels) | OpenTSLM-Cap-1B | 557 | 99.9\% | 59.21 | 60.18 | 61.12 | 60.56 | 0.515 |
| BMI (5 levels) | OpenTSLM-1B | 557 | 100.0\% | 59.47 | 60.70 | 62.30 | 61.50 | 0.517 |
| BMI (5 levels) | OpenTSLM-3B | 557 | 100.0\% | 56.31 | 57.62 | 59.19 | 58.59 | 0.526 |
| Blood Pressure (5 levels) | OpenTSLM-Cap-3B | 36 | 16.2\% | 27.52 | 8.05 | 33.21 | 9.85 | 0.972 |
| Blood Pressure (5 levels) | OpenTSLM-Cap-1B | 1 | 0.0\% | — | 0.00 | — | 0.00 | 1.000 |
| Blood Pressure (5 levels) | OpenTSLM-1B | 0 | 0.0\% | — | — | — | — | — |
| Blood Pressure (5 levels) | OpenTSLM-3B | 223 | 42.5\% | — | 0.00 | — | 0.00 | 1.063 |
| Happiness (4 levels) | OpenTSLM-Cap-3B | 328 | 100.0\% | 9.02 | 8.02 | 12.65 | 8.34 | 0.716 |
| Happiness (4 levels) | OpenTSLM-Cap-1B | 328 | 100.0\% | 7.79 | 7.54 | 8.71 | 4.00 | 0.726 |
| Happiness (4 levels) | OpenTSLM-1B | 328 | 100.0\% | 6.80 | 5.04 | 13.30 | 6.57 | 0.713 |
| Happiness (4 levels) | OpenTSLM-3B | 328 | 100.0\% | 4.70 | 3.00 | 5.42 | 0.80 | 0.787 |
| Sleep Time (4 levels) | OpenTSLM-Cap-3B | 29 | 10.1\% | — | 0.00 | — | 0.00 | 0.621 |
| Sleep Time (4 levels) | OpenTSLM-Cap-1B | 15 | 0.0\% | — | 0.00 | — | 0.00 | 0.733 |
| Sleep Time (4 levels) | OpenTSLM-1B | 7 | 0.0\% | — | 0.00 | — | 0.00 | 0.286 |
| Sleep Time (4 levels) | OpenTSLM-3B | 316 | 20.4\% | — | 0.00 | — | 0.00 | 0.563 |
