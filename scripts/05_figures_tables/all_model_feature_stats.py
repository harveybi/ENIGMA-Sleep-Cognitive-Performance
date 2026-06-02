import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')
import utils

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import julearn
import itertools
import starbars

from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
)
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

# %%
# ------------------------------- configuration -------------------------------
base_dir     = Path("/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results")

feature_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
                'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
                'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE',
                'Sleep_Cov_APOE_Shuffle', 'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle']

target_list  = ['Stroop', 'Memory', 'Stroop_rgo_age', 'Memory_rgo_age']

model_list   = ['Dummy', 'Linear', 'Ridge', 'SVM-linear', 'SVM-rbf',
                'rf', 'XGBoost', 'AutoGluon']

metrics      = ['test_r2',
                'test_neg_root_mean_squared_error',
                'test_neg_mean_absolute_error',
                'test_r_corr',
                'test_spearmanr']

# ------------------------------- collect results ------------------------------
records = {}                                          #  {(target, feature): {(model, metric): "µ (σ)"}}

for model, target, feature in itertools.product(model_list, target_list, feature_list):
    trend_folder = "SHIP_Trend_htcondor" if model == "XGBoost" else "SHIP_Trend"
    csv_path     = base_dir / model / target / trend_folder / feature / "scores.csv"
    if not csv_path.exists():
        print(f"\n❌ {csv_path} not found")
        continue                                      # skip missing combinations

    df = pd.read_csv(csv_path)

    for metric in metrics:
        if metric not in df.columns:
            print(f"\n❌ {metric} not found in {csv_path}")
            continue                                  # skip if that metric wasn’t saved

        mean, std = df[metric].mean(), df[metric].std()
        records.setdefault((target, feature), {})[(model, metric)] = f"{mean:.2f} ({std:.2f})"  # f"{mean:.3f} ({std:.3f})"

# ------------------------------- build tidy table -----------------------------
summary = (pd.DataFrame.from_dict(records, orient="index")
           .sort_index()                                      # alphabetical rows
           .sort_index(axis=1, level=[0, 1]))                 # Model → Metric

summary.index.names   = ["Target", "Feature"]
summary.columns.names = ["Model", "Metric"]

# --------- show in a notebook / interactive session (optional) ---------------
with pd.option_context("display.max_rows", None,
                       "display.max_columns", None,
                       "display.max_colwidth", None):
    print(summary)     # Jupyter / IPython

# %%
out_dir = Path(
    "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code"
    "/Results_making/model_feature_stats"
)
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2.  One CSV for each Target–Feature combo
#     • The row in `summary` is a Series whose index is (Model, Metric)
#     • We reshape it so that:  rows = Metrics,  columns = Models
# ---------------------------------------------------------------------------
for (target, feature), row in summary.iterrows():
    # reshape: metrics as rows, models as columns
    tidy = row.unstack(level="Model").sort_index()
    tidy.index.name = "Metric"        # nice header

    # safe, descriptive filename: e.g.  Stroop__Sleep_Cov.csv
    safe_target  = target.replace("/", "_")
    safe_feature = feature.replace("/", "_")
    fname = f"{safe_target}__{safe_feature}.csv"

    tidy.to_csv(out_dir / fname)

print(f"✅ {len(summary)} CSV files written to: {out_dir}")

# %%
metric_rename = {
    "test_neg_mean_absolute_error":     "Test MAE",
    "test_neg_root_mean_squared_error": "Test MSE",
    "test_r2":                          "Test R2",
    "test_r_corr":                      "Test Pearson r",
    "test_spearmanr":                   "Test Spearman r",
}
neg_metrics   = {"test_neg_mean_absolute_error",
                 "test_neg_root_mean_squared_error"}
model_rename  = {"rf": "Random forest"}

# final column order (after the rf→Random forest rename)
model_order = [
    "Dummy", "Linear", "Ridge", "SVM-linear", "SVM-rbf",
    "Random forest", "XGBoost", "AutoGluon"
]

def flip_mean_str(cell: str) -> str:
    """Flip the sign of the mean in a 'mean (std)' string; return unchanged if parsing fails."""
    if not isinstance(cell, str) or "(" not in cell:
        return cell
    mean_part, rest = cell.split("(", 1)
    try:
        mean_val = float(mean_part.strip())
    except ValueError:
        return cell
    std_part = rest.rstrip(") ").strip()
    return f"{-mean_val:.2f} ({std_part})"  # return f"{-mean_val:.3f} ({std_part})"

for csv_path in out_dir.glob("*.csv"):
    df = pd.read_csv(csv_path, index_col=0)

    # 1️⃣  sign flip for negative‑error metrics (skip XGBoost column)
    for metric in neg_metrics:
        if metric in df.index:
            for col in df.columns:
                if col != "XGBoost":
                    df.loc[metric, col] = flip_mean_str(df.loc[metric, col])

    # 2️⃣  friendly names
    df.rename(index=metric_rename,   inplace=True)
    df.rename(columns=model_rename, inplace=True)

    # 3️⃣  enforce metric row order (drop metrics that weren’t present)
    df = df.reindex(list(metric_rename.values())).dropna(how="all")

    # 4️⃣  enforce model column order (keep only those actually present)
    df = df.reindex(columns=[m for m in model_order if m in df.columns])

    # 5️⃣  save as *_revised.csv*
    df.to_csv(csv_path.with_name(f"{csv_path.stem}_revised.csv"))

print("✅ All CSVs cleaned, re‑ordered, and saved with '_revised.csv' suffix.")