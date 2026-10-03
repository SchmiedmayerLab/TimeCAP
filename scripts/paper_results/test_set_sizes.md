<!--
SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# Test-set composition

Per-task user counts on the captioned test partition. Binary tasks
report the minority-class user count (matching the AUPRC reporting
convention used in the paper). Ordinal tasks report the per-level
distribution; uneven distributions reduce the effective signal in
Spearman~ρ and QWK.


## Binary tasks

| Task | minority label | n\_users | n\_pos | prev (%) |
|---|---|---:|---:|---:|
| Biological Sex | Female | 587 | 149 | 25.4 |
| Hypertension | True | 279 | 90 | 32.3 |
| Employment | False | 531 | 127 | 23.9 |
| Sleep Diagnosis | True | 530 | 101 | 19.1 |
| Diabetes$^{\dagger}$ | True | 279 | 23 | 8.2 |
| Cardiovascular Disease$^{\dagger}$ | True | 464 | 71 | 15.3 |
| Coronary Artery Disease$^{\dagger}$ | True | 464 | 32 | 6.9 |
| Atrial Fibrillation$^{\dagger}$ | True | 464 | 15 | 3.2 |

## Ordinal tasks

| Task | n\_users | Level distribution (n, %) |
|---|---:|---|
| BMI (5 levels) | 557 | Underweight 25 (4.5%); Normal weight 175 (31.4%); Overweight 206 (37.0%); Obesity 121 (21.7%); Morbid Obesity 30 (5.4%) |
| Happiness (4 levels) | 328 | Low 28 (8.5%); Medium 53 (16.2%); High 133 (40.5%); Very High 114 (34.8%) |
| Blood Pressure (5 levels) | 277 | Normal 99 (35.7%); Elevated 56 (20.2%); Hypertension_Stage_1 101 (36.5%); Hypertension_Stage_2 21 (7.6%); Hypertensive_Crisis 0 (0.0%) |
| Sleep Time (4 levels) | 529 | Insufficient 49 (9.3%); Short 186 (35.2%); Normal 275 (52.0%); Too Long 19 (3.6%) |
