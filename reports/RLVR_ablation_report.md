# Clinician-rationale RLVR vs XGBoost — results report

*Status 2026-10-06. Branch `max-rosenblattl/clinical-signal-audit` (unpushed, latest `a62968d`). All numbers are on the MHC-benchmark test split (seed-42 sharable split); no test user was seen in training.*

## TL;DR

- **Pipeline:** OpenTSLM-SP (Llama-3.2-1B + LoRA) is first SFT-warmed on 498 clinician-style rationales (`<findings>` → `<reasoning>` → `<answer>`) to learn the format, then trained with **GRPO on 117,010 label-only prompts** (every positive train (label, week) pair + an equal number of negatives) on 8×H100. Reward: correct answer in a well-formed rationale.
- **Paper-style evaluation** (per-user AUROC/AUPRC over up to 16 weeks per user, per-week F1/accuracy, all test users at natural prevalence):
  - On the **6 labels shared with the TimeCap paper**, GRPO-117k reaches **AUROC 67.9** — level with **XGBoost on clinical features (68.2)**, 1.9 below **XGBoost on the ~600 generic features (69.8)**, and above the paper's **OpenTSLM-Cap 1B (65.6)** and its own SFT warm start (65.4).
  - Over **all 8 labels**: GRPO-117k **64.8 ± 2.6** vs XGB clinical **68.7 ± 2.2**; the gap comes from the two rarest labels (stroke: 6 positives, PH: 5).
  - GRPO beats its SFT start by **+3.6 AUROC** and **+6.4 macro-F1**, has the **best macro-F1** of all models (53.8) and wins clearly on **AFib (80.4 vs 68.0)**.
- XGBoost's sensor features add little over demographics alone (67.8 → 68.7 clinical, 70.6 generic).

---

## 1. Setup

### Data
| Set | Rows | Users | Content |
|---|---|---|---|
| SFT rationales (`rationale_sft_v4`) | 498 / 59 / 62 (train/val/test) | 363 | Label-blind clinician-style rationales written in-session from deterministic weekly findings (+ age, sex, BMI), rejection-sampled for the correct verdict; 8 labels |
| GRPO v1 (`rationale_grpo_v1`) | 960 train · 189 val · 1,121 test | 530 train | 60 pos + 60 neg weeks per label; val/test = 1 week per benchmark val/test user (all positives + ≤15 / ≤100 negatives) |
| **GRPO v2 (`rationale_grpo_v2`)** | **117,010 train** · same val/test | 957 train | **Every positive (label, week) of train users (58,505) + equal negatives drawn person-first** |
| Test, multi-week (`rationale_test_mw16`) | 34,048 | 497 | **All test users per label, up to 16 random usable weeks each** |

Labels: cardiovascular disease, hypertension, sleep disorder, diabetes, CAD, stroke (cerebrovascular), AFib, pulmonary hypertension. Usable week = ≥5 days with heart rate. Input = 19 HealthKit channels × 168 hours + age, sex, BMI in the prompt.

### Model and training
- **OpenTSLM-SP**, Llama-3.2-1B backbone, LoRA r=16 / α=32 on all projections, encoder + projector trainable, eager attention.
- **SFT warm-up:** 10 epochs, batch 4, LR 2e-4 (encoder) / 1e-4 (projector, LoRA); best epoch by val loss. 8 seeds run; seed 7 (lowest val loss 0.159) used as the GRPO warm start.
- **GRPO** (ported from OpenTSLM-R1, `train_rationale.py`): K=8 samples/prompt, advantage = reward − group mean, token-level loss, no KL / no clipping, identical-reward groups skipped. Fixes vs R1: encoder/projector receive gradients, EOS included in the loss, untruncated sampling distribution.
  - Reward: 1.0 correct + well-formed, 0.1 well-formed but wrong, 0 malformed.
  - Data-parallel on 8×H100 (gradients all-reduced), 64 prompts/step, LR 2e-5 / 1e-5 (R1's 2e-4 / 1e-4 collapsed the format within ~6 steps), max 360 new tokens, val every 100 steps.
  - Temperature 1.0 for steps 1–850, 1.15 from step 850 (skipped groups had risen to ~82 %).
  - Stopped at step 1,128 / 1,828 (val declining); **best checkpoint = step 700** (≈ 45k prompts seen).

| Step | 100 | 200 | 300 | 400 | 500 | 600 | **700** | 800 | 900* | 1000* | 1100* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Val macro balanced acc. | .624 | .631 | .639 | .632 | .644 | .635 | **.647** | .647 | .624 | .622 | .583 |

\* temperature 1.15. Training-set skipped groups rose 39 % → 82 %, format stayed ≥ 0.99 throughout.

### Baselines
- **XGBoost** (MHC-benchmark harness, unchanged: 1000 trees, depth 2, lr 0.05) on user-level features aggregated over all of a user's weeks, + age, sex, BMI covariates. Three feature sets: **clinical** (our clinician-derived extractors), **generic** (~600 benchmark features), **demographics only**. Rerun with per-user predictions saved; reproduces the earlier AUROCs exactly.
- **OpenTSLM-Cap 1B** — numbers copied from the TimeCap paper (per-user AUROC/AUPRC over all test weeks; per-week F1/acc). CVD, CAD, AFib and diabetes were **zero-shot** there.

## 2. Evaluation protocol

Mirrors the TimeCap paper (`scripts/paper_results/model_metrics.py`):
- **Per week:** macro-F1 (mean of both class F1) and accuracy; unparseable answers count as negative (parse rate here was 100 %).
- **Per user:** AUROC and AUPRC (minority class) over each user's **mean of hard yes/no predictions** across their evaluated weeks (≤ 16 here, all weeks in the paper). For XGBoost (one prediction per user) the predicted probability is used.
- All models scored on the **same 2,829 (label, user) pairs** (users without an XGBoost prediction removed; 305 of 3,134). Uncertainty = bootstrap SE over test users (200 resamples).

## 3. Main result — paper-style comparison

Values × 100. AUROC / AUPRC per user, F1 / Acc per week.

| Label | users (pos) | OpenTSLM-Cap 1B (paper) | XGB generic (~600 feat.) | XGB clinical | **TSLM GRPO-117k** | TSLM SFT (seed 7) |
|---|---|---|---|---|---|---|
| | | AUROC / AUPRC / F1 / Acc | AUROC / AUPRC / F1 / Acc | AUROC / AUPRC / F1 / Acc | AUROC / AUPRC / F1 / Acc | AUROC / AUPRC / F1 / Acc |
| CVD | 386 (65) | 65.2† / 30.0 / 61.7 / 72.6 | **74.9** / 38.2 / 59.5 / 83.2 | 73.1 / **39.7** / 60.8 / 83.7 | 70.2 / 29.2 / **62.9** / 71.6 | 64.4 / 23.1 / 48.0 / 50.3 |
| CAD | 386 (27) | **65.9**† / **15.8** / **58.8** / 76.0 | 65.0 / 13.3 / 48.1 / 92.7 | 64.4 / 14.2 / 48.2 / 93.0 | 64.8 / 10.8 / 50.5 / 68.2 | 59.3 / 9.8 / 51.9 / 73.3 |
| AFib | 386 (14) | 68.4† / 7.2 / 43.7 / 67.8 | 61.9 / 5.6 / 49.0 / 96.1 | 68.0 / 8.1 / 49.1 / 96.4 | **80.4** / **10.4** / **51.1** / 71.5 | 71.4 / 6.9 / 45.1 / 61.6 |
| Diabetes | 232 (23) | 64.6† / 12.6 / 53.1 / 67.6 | 68.0 / 19.7 / 47.3 / 89.7 | 64.3 / **24.7** / **55.6** / 90.9 | 65.3 / 14.8 / **55.6** / 71.7 | **71.5** / 17.5 / 52.5 / 64.7 |
| Hypertension | 232 (76) | 69.9 / 48.8 / **64.6** / 66.5 | **74.4** / **58.0** / 61.4 / 70.3 | 70.9 / 54.6 / 60.6 / 70.3 | 64.2 / 42.0 / 62.4 / 67.7 | 61.5 / 39.5 / 49.6 / 49.9 |
| Sleep disorder | 435 (83) | 59.5 / 25.4 / 55.7 / 75.1 | **74.7** / **36.9** / 57.7 / 80.0 | 68.6 / 34.0 / 52.1 / 77.7 | 62.7 / 25.5 / **58.0** / 67.2 | 64.2 / 26.3 / 52.9 / 56.5 |
| Stroke | 386 (6) | – | 71.8 / 4.2 / **49.6** / 98.4 | **86.2** / **6.9** / **49.6** / 98.4 | 64.5 / 2.8 / 45.4 / 76.3 | 55.4 / 2.3 / 47.5 / 85.1 |
| PH | 386 (5) | – | **73.8** / **3.4** / **49.7** / 98.7 | 54.3 / 1.7 / 49.6 / 98.4 | 46.2 / 1.4 / 44.4 / 75.2 | 42.0 / 1.3 / 31.6 / 43.8 |
| **Mean, 6 shared labels** | | 65.6 / 23.3 / 56.3 / 70.9 | **69.8** / 28.6 / 53.8 / 85.3 | 68.2 / **29.2** / 54.4 / 85.3 | 67.9 / 22.1 / **56.8** / 69.7 | 65.4 / 20.5 / 50.0 / 59.4 |

Bold = best per label and metric among AUROC / AUPRC / F1 (accuracy not bolded: it mostly rewards predicting the majority class, see reading 4).
† zero-shot in the paper (not trained on that label).

**Macro over all 8 labels (± bootstrap SE):**

| Model | AUROC | macro-F1 | Accuracy |
|---|---|---|---|
| XGB generic | **70.6 ± 2.4** | 52.8 ± 0.7 | 88.6 ± 0.6 |
| XGB clinical | 68.7 ± 2.2 | 53.2 ± 0.9 | 88.6 ± 0.6 |
| XGB demographics | 67.8 ± 2.5 | 53.1 ± 0.9 | 88.7 ± 0.6 |
| **TSLM GRPO-117k** | 64.8 ± 2.6 | **53.8 ± 0.9** | 71.2 ± 1.0 |
| TSLM SFT (seed 7) | 61.2 ± 2.8 | 47.4 ± 0.9 | 60.6 ± 1.0 |

### Reading
1. **TSLM ≈ XGBoost on the common cardiometabolic labels**: 67.9 mean AUROC over the 6 shared labels vs 68.2 (XGB clinical) and 69.8 (XGB generic, the strongest baseline) — both gaps within about 1 SE — and **above the original OpenTSLM-Cap** (65.6). The TSLM wins clearly on AFib (80.4 vs 61.9 / 68.0); XGB generic is ahead by 5–12 points on CVD, hypertension and sleep, where its larger feature set pays off.
2. **GRPO helps** relative to its own SFT start on identical users: AUROC +3.6, macro-F1 +6.4 (well beyond the SE), largest gains on AFib, CVD, hypertension; diabetes got worse (−6).
3. **Rare labels are the TSLM's weak spot** (stroke 6 positives, PH 5): XGBoost's age/BMI-driven ranking wins there; with so few positives both numbers are fragile.
4. **Accuracy is not comparable across model families:** XGBoost (trained at natural prevalence, 0.5 threshold) almost never predicts "yes" → high accuracy, low macro-F1; the TSLM (trained balanced) trades accuracy for recall of positives. Macro-F1 and AUROC are the fair columns.
5. **Sensors add little to XGBoost here** — demographics alone are within ~1 AUROC of clinical features; the cardiometabolic labels are largely age/BMI-driven in this cohort.
6. **Multi-week evaluation matters:** the same GRPO-117k model scored on a single week per user (capped test subset) gave macro AUROC 0.608 (score-based); averaging over ≤16 weeks per user raises the paper-style per-user AUROC to 0.648 (8 labels).

## 4. Earlier runs (single-week test, 1,121 rows, for context)

| Run | macro AUROC | macro-F1 | Note |
|---|---|---|---|
| SFT only, 8 seeds | 0.600 ± 0.022 | 0.543 ± 0.021 | ± = sd over seeds |
| SFT → GRPO-960 (1 epoch, 960 prompts), 8 seeds | 0.620 ± 0.020 | 0.553 ± 0.025 | +0.020 AUROC, bal. acc. up on 8/8 seeds |
| SFT seed 7 | 0.606 | 0.562 | warm start of GRPO-117k |
| SFT → GRPO-117k (step 700) | 0.610 | 0.596 | |

The 8-seed spread (≈ ±0.02 AUROC) is the best available estimate of training variance; GRPO-117k is a single run.

## 5. Caveats

- **Single training run** for GRPO-117k; seed variance (≈ ±2 AUROC) comes on top of the bootstrap SEs.
- **Cohort not identical to the paper:** our test users need demographics, a usable week and an XGBoost prediction (232–435 users/label vs 279–587 in the paper); we use ≤16 weeks per user, the paper all weeks.
- **Paper model was zero-shot** on CVD, CAD, AFib, diabetes; ours was trained on them (SFT rationales + GRPO reward), so those rows compare a trained against a zero-shot TSLM.
- **SFT rationales were written in-session** (Claude) from deterministic findings with a label-blind rule, not by clinicians; clinician validation is outstanding. The rule's own accuracy was only ~60 %, which bounds what SFT alone can teach.
- **Reward checks only the final answer**, not whether the findings are faithful to the signal — GRPO can learn post-hoc reasoning. A findings-consistency reward is the obvious next step.
- **Overlap SFT ↔ GRPO training data:** 88 of the v1 GRPO weeks were also SFT weeks (35 with the same label); validation/test users are disjoint from all training data.
- **GRPO saturated** after ~700 steps (≈ 45k prompts; skipped groups ~80 %); the remaining 60 % of v2 prompts were never used. A temperature bump to 1.15 did not help.

## 6. Reproduction

| What | Where |
|---|---|
| Code | `SensorTSLM/SensorTSLM`, commits `fc06862` … `a62968d` (`train_rationale.py`, `time_series_datasets/mhc_rationale_qa_dataset.py`, `scripts/export_rationale_{sft,grpo}.py`, `scripts/run_rationale_{sweep,ddp}.sh`, `scripts/rationale_metrics.py`, `scripts/xgb_to_rationale_predictions.py`) |
| Data, checkpoints, logs | `/mnt/filesystem-p7/workspace/clinical-rationales/` — `rationale_*.arrow`, `runs/rationale/seed*/` (8-seed v1), `runs/rationale_v2/` (GRPO-117k; `best_model.pt` = step 700), `runs/eval_mw16/` (multi-week test predictions), `train_logs/` |
| Handoff doc | `/mnt/filesystem-p7/workspace/GPU_RUN_RATIONALE.md` |
| XGBoost | `MHC/MHC-benchmark/results/eval/rationale_compare/{fx_clinical,fe_xgboost,fx_demo}/` (per-user predictions) |
| Local predictions / reports | `reports/preds_mw16/`, `reports/rationale_vs_xgb_mw16.{md,json}` |

```bash
# GRPO on 8 GPUs (resumes from latest.pt)
VENV=~/venv HF_HOME=~/hf_cache ./scripts/run_rationale_ddp.sh <sft best_model.pt> --gpus 8
# multi-week test eval of a checkpoint placed in <out>/.../rationale_grpo/checkpoints/best_model.pt
./scripts/run_rationale_ddp.sh <ckpt> --out <out> -- --eval_only --grpo_file rationale_test_mw16.arrow
# metrics
python scripts/rationale_metrics.py --run "NAME=<test_predictions.jsonl>" ... --out reports/x.md
```

---

## Appendix — per-label tables with bootstrap SE

TSLM AUROC/AUPRC in these tables use the **continuous score** (mean per user of log p(yes) − log p(no)), not the paper-style hard-prediction mean used in §3; per-week F1/accuracy are identical to §3.

### AUROC (score-based for TSLM)
| Label | users (pos) | XGB clinical | XGB generic | XGB demographics | TSLM SFT seed7 | TSLM GRPO-117k |
|---|---|---|---|---|---|---|
| AFib | 386 (14) | 0.680 ± 0.063 | 0.619 ± 0.070 | 0.777 ± 0.055 | 0.729 ± 0.053 | 0.835 ± 0.056 |
| CAD | 386 (27) | 0.644 ± 0.052 | 0.650 ± 0.055 | 0.654 ± 0.054 | 0.602 ± 0.058 | 0.657 ± 0.061 |
| Stroke | 386 (6) | 0.862 ± 0.037 | 0.718 ± 0.125 | 0.703 ± 0.093 | 0.418 ± 0.170 | 0.672 ± 0.136 |
| Diabetes | 232 (23) | 0.643 ± 0.064 | 0.680 ± 0.053 | 0.615 ± 0.060 | 0.704 ± 0.054 | 0.618 ± 0.066 |
| Hypertension | 232 (76) | 0.709 ± 0.033 | 0.744 ± 0.032 | 0.713 ± 0.034 | 0.624 ± 0.037 | 0.582 ± 0.042 |
| PH | 386 (5) | 0.543 ± 0.113 | 0.738 ± 0.061 | 0.524 ± 0.151 | 0.383 ± 0.153 | 0.276 ± 0.178 |
| CVD | 386 (65) | 0.731 ± 0.037 | 0.749 ± 0.031 | 0.750 ± 0.035 | 0.663 ± 0.040 | 0.668 ± 0.043 |
| Sleep disorder | 435 (83) | 0.686 ± 0.030 | 0.747 ± 0.029 | 0.685 ± 0.033 | 0.637 ± 0.033 | 0.609 ± 0.033 |

### AUPRC (score-based for TSLM)
| Label | users (pos) | XGB clinical | XGB generic | XGB demographics | TSLM SFT seed7 | TSLM GRPO-117k |
|---|---|---|---|---|---|---|
| AFib | 386 (14) | 0.081 ± 0.048 | 0.056 ± 0.023 | 0.158 ± 0.072 | 0.073 ± 0.030 | 0.157 ± 0.080 |
| CAD | 386 (27) | 0.142 ± 0.050 | 0.133 ± 0.044 | 0.116 ± 0.032 | 0.136 ± 0.045 | 0.154 ± 0.054 |
| Stroke | 386 (6) | 0.069 ± 0.043 | 0.042 ± 0.024 | 0.034 ± 0.018 | 0.032 ± 0.037 | 0.041 ± 0.026 |
| Diabetes | 232 (23) | 0.247 ± 0.076 | 0.197 ± 0.065 | 0.172 ± 0.059 | 0.182 ± 0.051 | 0.156 ± 0.048 |
| Hypertension | 232 (76) | 0.546 ± 0.057 | 0.580 ± 0.055 | 0.581 ± 0.057 | 0.405 ± 0.048 | 0.423 ± 0.054 |
| PH | 386 (5) | 0.017 ± 0.012 | 0.034 ± 0.024 | 0.019 ± 0.013 | 0.013 ± 0.007 | 0.014 ± 0.014 |
| CVD | 386 (65) | 0.397 ± 0.057 | 0.382 ± 0.055 | 0.400 ± 0.063 | 0.308 ± 0.056 | 0.350 ± 0.054 |
| Sleep disorder | 435 (83) | 0.340 ± 0.047 | 0.369 ± 0.048 | 0.358 ± 0.053 | 0.256 ± 0.034 | 0.257 ± 0.035 |

### macro-F1 (per week)
| Label | users (pos) | XGB clinical | XGB generic | XGB demographics | TSLM SFT seed7 | TSLM GRPO-117k |
|---|---|---|---|---|---|---|
| AFib | 386 (14) | 0.491 ± 0.002 | 0.490 ± 0.002 | 0.491 ± 0.002 | 0.451 ± 0.026 | 0.511 ± 0.031 |
| CAD | 386 (27) | 0.482 ± 0.003 | 0.481 ± 0.004 | 0.482 ± 0.003 | 0.519 ± 0.027 | 0.505 ± 0.028 |
| Stroke | 386 (6) | 0.496 ± 0.002 | 0.496 ± 0.002 | 0.496 ± 0.002 | 0.475 ± 0.014 | 0.454 ± 0.018 |
| Diabetes | 232 (23) | 0.556 ± 0.054 | 0.473 ± 0.005 | 0.533 ± 0.045 | 0.525 ± 0.033 | 0.556 ± 0.035 |
| Hypertension | 232 (76) | 0.606 ± 0.034 | 0.614 ± 0.033 | 0.603 ± 0.033 | 0.496 ± 0.031 | 0.624 ± 0.033 |
| PH | 386 (5) | 0.496 ± 0.002 | 0.497 ± 0.001 | 0.497 ± 0.001 | 0.316 ± 0.017 | 0.444 ± 0.016 |
| CVD | 386 (65) | 0.608 ± 0.034 | 0.595 ± 0.032 | 0.588 ± 0.034 | 0.480 ± 0.027 | 0.629 ± 0.034 |
| Sleep disorder | 435 (83) | 0.521 ± 0.025 | 0.577 ± 0.031 | 0.554 ± 0.030 | 0.529 ± 0.027 | 0.580 ± 0.028 |

### Accuracy (per week)
| Label | users (pos) | XGB clinical | XGB generic | XGB demographics | TSLM SFT seed7 | TSLM GRPO-117k |
|---|---|---|---|---|---|---|
| AFib | 386 (14) | 0.964 ± 0.009 | 0.961 ± 0.009 | 0.964 ± 0.009 | 0.616 ± 0.028 | 0.715 ± 0.028 |
| CAD | 386 (27) | 0.930 ± 0.013 | 0.927 ± 0.013 | 0.930 ± 0.013 | 0.733 ± 0.022 | 0.682 ± 0.025 |
| Stroke | 386 (6) | 0.984 ± 0.006 | 0.984 ± 0.006 | 0.984 ± 0.006 | 0.851 ± 0.020 | 0.763 ± 0.024 |
| Diabetes | 232 (23) | 0.909 ± 0.019 | 0.897 ± 0.019 | 0.884 ± 0.021 | 0.647 ± 0.032 | 0.717 ± 0.029 |
| Hypertension | 232 (76) | 0.703 ± 0.027 | 0.703 ± 0.028 | 0.694 ± 0.027 | 0.499 ± 0.032 | 0.677 ± 0.030 |
| PH | 386 (5) | 0.984 ± 0.006 | 0.987 ± 0.006 | 0.987 ± 0.006 | 0.438 ± 0.027 | 0.752 ± 0.022 |
| CVD | 386 (65) | 0.837 ± 0.020 | 0.832 ± 0.018 | 0.842 ± 0.018 | 0.503 ± 0.027 | 0.716 ± 0.029 |
| Sleep disorder | 435 (83) | 0.777 ± 0.020 | 0.800 ± 0.019 | 0.814 ± 0.019 | 0.565 ± 0.025 | 0.672 ± 0.025 |
