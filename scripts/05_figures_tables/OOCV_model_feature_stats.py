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

from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
)
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

# %%
# ------------------------------- configuration -------------------------------
base_dir = Path("/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results")

site_list = ['EMC', 'Juelich', 'KI', 'Liege', 'Pitts', 'VETSA']

target_list_EMC = ['Stroop', 'Stroop_rgo_age']
target_list_KI = ['Memory', 'Memory_rgo_age']
target_list_Liege = ['Stroop', 'Memory', 'Stroop_rgo_age', 'Memory_rgo_age']
target_list_Pitts = ['Executive', 'Memory_Letter', 'Memory_Spatial', 'Executive_rgo_age', 'Memory_Letter_rgo_age', 'Memory_Spatial_rgo_age']
target_list_VETSA = ['Stroop', 'Memory_Digit', 'Memory_Letter', 'Stroop_rgo_age', 'Memory_Digit_rgo_age', 'Memory_Letter_rgo_age']

session_list_Juelich = ['sess-1', 'sess-2', 'sess-SD']
raw_target_list_Juelich = ['Memory_Letter', 'Memory_Letter_rgo_age', 'Memory_Spatial', 'Memory_Spatial_rgo_age']

target_list_Juelich = []
for session in session_list_Juelich:
    for target in raw_target_list_Juelich:
        target_list_Juelich.append(f"{session}/{target}")

feature_list_full = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
                     'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
                     'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE',
                     'Sleep_Cov_APOE_Shuffle', 'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle']

model_list = ['XGBoost', 'AutoGluon']

metrics = ['Performance Test Set MAE',
           'Performance Test Set RMSE',
           'Performance Test Set R2',
           'Performance Test Set Pearson r',
           'Performance Test Set Pearson r p-value',
           'Performance Test Set Spearman r',
           'Performance Test Set Spearman r p-value']

# %%
site_targets = {
    "EMC":     target_list_EMC,
    "Juelich": target_list_Juelich,
    "KI":      target_list_KI,
    "Liege":   target_list_Liege,
    "Pitts":   target_list_Pitts,
    "VETSA":   target_list_VETSA,
}

feature_rename_in_table = {
    "Sleep": "Sleep",
    "Cov": "Demographic",
    "Sleep_Cov": "Sleep, Demo",
    "Sleep_Shuffle_Cov": "Sleep (Shuffled), Demo",
    "Brain": "Brain",
    "Sleep_Cov_Brain": "Sleep, Demo, Brain",
    "Sleep_Cov_Brain_Shuffle": "Sleep (Shuffled), Demo, Brain",
    "Sleep_APOE": "Sleep, APOE",
    "Sleep_APOE_Shuffle": "Sleep (Shuffled), APOE",
    "Sleep_Cov_APOE": "Sleep, Demo, APOE",
    "Sleep_Cov_APOE_Shuffle": "Sleep (Shuffled), Demo, APOE",
    "Sleep_Cov_Brain_APOE": "Sleep, Demo, Brain, APOE",
    "Sleep_Cov_Brain_APOE_Shuffle": "Sleep (Shuffled), Demo, Brain, APOE"
}

feature_order_friendly = [
    feature_rename_in_table.get(f, f) for f in feature_list_full
]

# ---------------------------------------------------------------------------
# 2.  helper: load one metrics.csv and return a 1‑row dataframe
# ---------------------------------------------------------------------------
def read_metrics(csv_path: Path, site: str) -> pd.Series:
    """
    Read the metrics.csv (or whatever file name you use) and pull out the
    required columns, returning a Series with the expected final names:
        'Performance Test Set MAE <SITE>', ...
    """
    metrics_renamed = []
    # rename the metrics list to include the site name
    #   e.g. 'Performance Test Set MAE' → 'Performance Test Set MAE <site>'
    for m in metrics:
        new_name = f"{m} {site}"
        metrics_renamed.append(new_name)

    df = pd.read_csv(csv_path)          # ← adjust if your file is TSV, etc.
    # Assume each file has ONE row (aggregated over repeats/folds).
    row = df.iloc[0][metrics_renamed]           # keep only the desired columns
    return row

# ---------------------------------------------------------------------------
# 3.  main loop over sites
# ---------------------------------------------------------------------------
for site in site_list:
    # collect rows separately for each target
    rows_by_target = {t: [] for t in site_targets[site]}

    # traverse the directory tree once
    for model, feature, target in itertools.product(model_list,
                                                    feature_list_full,
                                                    site_targets[site]):
        metrics_csv = (base_dir / model / site / target / feature /
                       "results_metrics.csv")
        if not metrics_csv.exists():
            # uncomment to debug missing files
            # print("missing:", metrics_csv)
            continue

        s = read_metrics(metrics_csv, site)
        # rows_by_target[target].append({
        #     "Feature Combination": feature,
        #     "Model":               model,
        #     **s.to_dict()                       # metric columns
        # })
        rows_by_target[target].append({
            "Model": model,
            "Feature Combination": feature,
            **s.to_dict()  # metric columns
        })

    # ---------------------------------------------------------------
    # write one CSV per target
    # ---------------------------------------------------------------
    for target, rows in rows_by_target.items():
        if not rows:                      # nothing collected for this target
            print(f"{site} / {target}: no rows found")
            continue

        # df = (pd.DataFrame(rows)
        #         .sort_values(["Feature Combination", "Model"],
        #                      kind="stable")
        #         .set_index(["Feature Combination", "Model"]))
        df = (pd.DataFrame(rows).set_index(["Model", "Feature Combination"]))

        df = (pd.DataFrame(rows)
              .assign(**{
            "Feature Combination": lambda d:
            d["Feature Combination"]
                      .map(feature_rename_in_table)
                      .fillna(d["Feature Combination"])
        })
              .astype({
            "Feature Combination": pd.CategoricalDtype(
                categories=feature_order_friendly, ordered=True)
        })
              .set_index(["Model", "Feature Combination"])
              .sort_index(level=["Model", "Feature Combination"],
                          sort_remaining=False,
                          kind="stable"))

        df = df.rename(columns=lambda c: c.replace(f" {site}", ""))

        # ---------- add significance stars to the two 'r' columns ------------------
        def stars(p: float) -> str:
            if pd.isna(p):
                return ""
            if p <= 1e-4:
                return "****"
            elif p <= 1e-3:
                return "***"
            elif p <= 1e-2:
                return "**"
            elif p <= 5e-2:
                return "*"
            else:
                return ""


        def format_r_with_stars(r: float, p: float) -> str:
            if pd.isna(r):
                return ""
            return f"{r:.2f}{stars(p)}"  # two decimals + stars


        # annotate Pearson r
        df["Performance Test Set Pearson r"] = df.apply(
            lambda row: format_r_with_stars(
                row["Performance Test Set Pearson r"],
                row["Performance Test Set Pearson r p-value"]
            ),
            axis=1
        )

        # annotate Spearman r
        df["Performance Test Set Spearman r"] = df.apply(
            lambda row: format_r_with_stars(
                row["Performance Test Set Spearman r"],
                row["Performance Test Set Spearman r p-value"]
            ),
            axis=1
        )

        # drop the p-value columns
        df = df.drop(columns=[
            "Performance Test Set Pearson r p-value",
            "Performance Test Set Spearman r p-value"
        ])

        out_csv = base_dir / site / f"{target}_metrics_summary.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # df.to_csv(out_csv)
        df.to_csv(out_csv, float_format="%.2f")
        print(f"{site} / {target}: {len(df)} rows → {out_csv}")
