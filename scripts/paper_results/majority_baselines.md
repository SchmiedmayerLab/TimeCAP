# Majority-class baselines

Always-predict-majority on the held-out test split. AUROC is 0.5 by
definition for any constant scorer; AUPRC reduces to the per-user
positive-class prevalence. Spearman~ρ and QWK are degenerate (≈ 0) for
a constant ordinal predictor.


## Binary (per-week F1 / accuracy; per-user AUROC / AUPRC)

| Task | n_wk | n_users | wk-prev | user-prev | F1 | Acc | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Biological Sex | 41,162 | 587 | 0.769 | 0.746 | 86.95 | 76.91 | 50.00 | 25.38 |
| Hypertension | 25,711 | 279 | 0.416 | 0.323 | 0.00 | 58.37 | 50.00 | 32.26 |
| Employment | 37,790 | 531 | 0.716 | 0.761 | 83.43 | 71.57 | 50.00 | 23.92 |
| Sleep Diagnosis | 37,776 | 530 | 0.208 | 0.191 | 0.00 | 79.25 | 50.00 | 19.06 |
| Diabetes | 25,711 | 279 | 0.114 | 0.082 | 0.00 | 88.63 | 50.00 | 8.24 |
| Cardiovascular Disease | 34,560 | 464 | 0.231 | 0.153 | 0.00 | 76.94 | 50.00 | 15.30 |
| Coronary Artery Disease | 34,560 | 464 | 0.110 | 0.069 | 0.00 | 89.02 | 50.00 | 6.90 |
| Atrial Fibrillation | 34,560 | 464 | 0.030 | 0.032 | 0.00 | 97.02 | 50.00 | 3.23 |

## Ordinal (per-week QWK / MAE; per-user QWK)

| Task | n_wk | majority | QWK | MAE |
|---|---:|---|---:|---:|
| BMI (5 levels) | 37,510 | Overweight | 0.00 | 0.759 |
| Happiness (4 levels) | 28,610 | High | 0.00 | 0.665 |
| Blood Pressure (5 levels) | 25,479 | Hypertension_Stage_1 | 0.00 | 0.960 |
| Sleep Time (4 levels) | 37,220 | Normal | 0.00 | 0.593 |
