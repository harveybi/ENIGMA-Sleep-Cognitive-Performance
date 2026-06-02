import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')
import utils

from pathlib import Path
from collections import defaultdict

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

import seaborn as sns
sns.set_context("paper")

import matplotlib as mpl
# Tell Matplotlib: “Whenever I ask for sans-serif, try Arial first”
mpl.rcParams['font.family']      = 'sans-serif'
mpl.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

# %%
base_dir = Path("/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results")

main_results_feature_list = ['Sleep', 'Sleep_APOE', 'Sleep_APOE_Shuffle',
                             'Sleep_Cov', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle',
                             'Sleep_Cov_Brain', 'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle']
model_list = ['XGBoost', 'AutoGluon']
target_list = ['Stroop', 'Memory']

metrics_box_plot = ['test_r2', 'test_r_corr']

dfs = defaultdict(list)      # key = (model, target) → list of dataframes

rename_metrics = {           # nicer column names (optional)
    "test_r2":   "R2",
    "test_r_corr": "Pearson r",
}

for model, target, feature in itertools.product(model_list, target_list, main_results_feature_list):
    trend_folder = "SHIP_Trend_htcondor" if model == "XGBoost" else "SHIP_Trend"
    csv_path     = base_dir / model / target / trend_folder / feature / "scores.csv"
    if not csv_path.exists():
        print(f"\n {csv_path} not found")
        continue

    # ── load the CV rows and keep only the desired metrics ────────────────
    df = (pd.read_csv(csv_path, usecols=metrics_box_plot)
          .rename(columns=rename_metrics)
          .assign(Feature=feature))  # tag rows by feature

    dfs[(model, target)].append(df)

xgboost_Stroop = pd.concat(dfs[("XGBoost",   "Stroop")],  ignore_index=True)
xgboost_Memory = pd.concat(dfs[("XGBoost",   "Memory")],  ignore_index=True)
autogluon_Stroop = pd.concat(dfs[("AutoGluon", "Stroop")],  ignore_index=True)
autogluon_Memory = pd.concat(dfs[("AutoGluon", "Memory")],  ignore_index=True)

# %%
plot_dfs = {
    ("XGBoost",   "Stroop"):  xgboost_Stroop,
    ("XGBoost",   "Memory"):  xgboost_Memory,
    ("AutoGluon", "Stroop"):  autogluon_Stroop,
    ("AutoGluon", "Memory"):  autogluon_Memory,
}

# --------------------------------------------
# 2.  colour map for the swarm points
# --------------------------------------------
model_colour = {
    "XGBoost":   "#fd7351",
    "AutoGluon": "#fd2723",
}

metrics_to_plot = ["R2", "Pearson r"]          # columns in your dataframes

# --------------------------------------------
# 3.  loop and plot
# --------------------------------------------
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

for (model, target), df in plot_dfs.items():
    col = model_colour[model]

    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 8))

        # grey box‑plot frame
        sns.boxplot(data=df,
                    x="Feature", y=metric,
                    boxprops={"facecolor": 'none'},
                    whiskerprops={"color":"black"},
                    medianprops={"color":"black"},
                    capprops={"color":"black"},
                    showfliers=False)

        # coloured swarm overlay
        sns.swarmplot(data=df,
                      x="Feature", y=metric,
                      size=3,  linewidth=0,
                      color=col)

        plt.title(f"{model} – {target} – {metric}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()