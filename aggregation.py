import glob, os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BINARY = ["BiologicalSex", "Diabetes", "Hypertension",
          "cardiovascular_disease", "sleep_diagnosis1", "work"]

for task in BINARY:
    paths = sorted(glob.glob(
        f"/workspace/SensorTSLM/results/{task}/opentslm-best/2026*/predictions.csv"))
    if not paths:
        print(f"=== {task} === no joint run"); continue
    path = max(paths, key=os.path.getsize)
    df = pd.read_csv(path).dropna(subset=["ground_truth", "prediction", "user_id"])
    gt_vals = df["ground_truth"].unique()
    if len(gt_vals) != 2:
        print(f"=== {task} === gt!=2 classes: {list(gt_vals)[:5]}"); continue
    pos = df["ground_truth"].value_counts().idxmin()
    df["gt"]   = (df["ground_truth"] == pos).astype(int)
    df["pred"] = (df["prediction"]   == pos).astype(int)
    g = df.groupby("user_id").agg(gt=("gt","max"), pred=("pred","mean"))
    if g["gt"].nunique() < 2:
        print(f"=== {task} === single class at user level"); continue
    auroc = roc_auc_score(g["gt"], g["pred"])
    auprc = average_precision_score(g["gt"], g["pred"])
    print(f"=== {task} (pos={pos!r}) ===")
    print(f"  rows={len(df)}  users={len(g)}  "
          f"pos_rate_row={df['gt'].mean():.3f}  pos_rate_user={g['gt'].mean():.3f}")
    print(f"  per-user AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    print(f"  ({os.path.basename(os.path.dirname(path))})")

