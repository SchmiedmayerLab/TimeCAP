<!--

This source file is part of the TimeCAP open-source project

SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Majority-class baselines

Always-predict-majority on the held-out test split. AUROC is 0.5 by
definition for any constant scorer; AUPRC reduces to the per-user
positive-class prevalence. Spearman~ρ and QWK are degenerate (≈ 0) for
a constant ordinal predictor.


## Binary (per-week macro-F1 / accuracy; per-user AUROC / AUPRC)

| Task | n_wk | n_users | n_pos | minority-prev | F1$_{+}$ | macro-F1 | Acc | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Biological Sex | 41,162 | 587 | 149 | 0.254 | 86.95 | 43.47 | 76.91 | 50.00 | 25.38 |
| Hypertension | 25,711 | 279 | 90 | 0.323 | 0.00 | 36.86 | 58.37 | 50.00 | 32.26 |
| Employment | 37,790 | 531 | 127 | 0.239 | 83.43 | 41.71 | 71.57 | 50.00 | 23.92 |
| Sleep Diagnosis | 37,776 | 530 | 101 | 0.191 | 0.00 | 44.21 | 79.25 | 50.00 | 19.06 |
| Diabetes | 25,711 | 279 | 23 | 0.082 | 0.00 | 46.99 | 88.63 | 50.00 | 8.24 |
| Cardiovascular Disease | 34,560 | 464 | 71 | 0.153 | 0.00 | 43.49 | 76.94 | 50.00 | 15.30 |
| Coronary Artery Disease | 34,560 | 464 | 32 | 0.069 | 0.00 | 47.10 | 89.02 | 50.00 | 6.90 |
| Atrial Fibrillation | 34,560 | 464 | 15 | 0.032 | 0.00 | 49.24 | 97.02 | 50.00 | 3.23 |

## Ordinal (per-week QWK / MAE; per-user QWK)

| Task | n_wk | n_users | majority | QWK | MAE |
|---|---:|---:|---|---:|---:|
| BMI (5 levels) | 37,510 | 557 | Overweight | 0.00 | 0.759 |
| Happiness (4 levels) | 28,610 | 328 | High | 0.00 | 0.665 |
| Blood Pressure (5 levels) | 25,479 | 277 | Hypertension_Stage_1 | 0.00 | 0.960 |
| Sleep Time (4 levels) | 37,220 | 529 | Normal | 0.00 | 0.593 |
