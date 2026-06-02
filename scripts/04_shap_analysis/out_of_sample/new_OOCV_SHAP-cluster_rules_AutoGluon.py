import os
import sys
from pathlib import Path
from typing import Optional

# ==========================
# Config (no argparse)
# ==========================
# Only work on Sleep_Cov
feature_comb = "Sleep_Cov"   # fixed
# Choose target here manually
target = "Stroop"            # "Stroop", "Memory"

print(f"\nStarting SHAP-IQ OOCV clustering → target={target}, feature_comb={feature_comb}\n")

# ==========================
# Imports
# ==========================
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils  # noqa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")

import shap
import shapiq
from shapiq.interaction_values import aggregate_interaction_values
import pickle

from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.utils.validation import check_array

import collections, collections.abc, six, sklearn
collections.Iterable = collections.abc.Iterable
sklearn.externals.six = six
from skrules import SkopeRules
from cluster_explorer import Explainer as CE_explainer  # noqa: F401

import umap.umap_ as umap
from joblib import dump, load  # noqa: F401

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib as mpl
mpl.rcParams['font.family']     = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})

# ===========
# Plot constants & helpers
# ===========
FS_TICK  = 12
FS_LABEL = 12
FS_TITLE = 14

# Network plots: close to square; other plots rectangular
DEFAULT_FIGSIZE_SQ = (3.6, 3.0)  # 6, 5; 5.4, 4.5; 4.8, 4.0
DEFAULT_FIGSIZE_RX = (6.0, 4.0)

SHOW_PLOTS = True  # default; functions have own `show` arg

LABEL_REPLACEMENTS = {
    "Age_at_Scan": "Age",
    "Self_Sleep_Dur": "Self Sleep Duration",
    "Self_Sleep_Eff": "Self Sleep Efficiency",
    "PSG_Sleep_Dur": "PSG Sleep Duration",
    "PSG_Sleep_Eff": "PSG Sleep Efficiency",
    "Depression_score": "Depressive Score",
}

BASE_COLORS = [
    "#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
    "#984ea3", "#999999", "#e41a1c", "#dede00"
]


def pretty_names(names: list[str]) -> list[str]:
    return [LABEL_REPLACEMENTS.get(n, n) for n in names]


def style_axes(ax):
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=FS_TITLE)
    return ax


def maybe_save_show(
    outpath: Optional[str],
    show: bool = SHOW_PLOTS,
    dpi: int = 300,
    close: bool = True,
):
    fig = plt.gcf()

    if outpath:
        p = Path(outpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        suf = p.suffix.lower()

        if suf == ".png":
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        elif suf == ".svg":
            fig.savefig(p, bbox_inches="tight")
        elif suf == "":
            fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
            fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        else:
            fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show or not outpath:
        plt.show()
    if close:
        plt.close()


# ==========================
# Feature lists
# ==========================
feature_lists_path = (
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/'
    'Code/Out-of-sample_validation/feature_lists.pkl'
)
with open(feature_lists_path, 'rb') as f:
    feature_lists = pickle.load(f)

Sleep              = feature_lists[0]
Cov                = feature_lists[1]
APOE4              = feature_lists[2]
TIV                = feature_lists[3]
Thickness_DK       = feature_lists[4]
Thickness_Schaefer = feature_lists[5]
Area_DK            = feature_lists[6]
Area_Schaefer      = feature_lists[7]
Subcortical        = feature_lists[8]
targets            = feature_lists[9]  # noqa: F841

sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']

X_dict = {
    'Sleep': Sleep,
    'Cov': Cov,
    'Sleep_Cov': Sleep + Cov,
    'Sleep_Shuffle_Cov': Sleep + Cov,
    'Brain': Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_Cov_Brain': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_Cov_Brain_Shuffle': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_APOE': Sleep + APOE4,
    'Sleep_APOE_Shuffle': Sleep + APOE4,
    'Sleep_Cov_APOE': Sleep + Cov + APOE4,
    'Sleep_Cov_APOE_Shuffle': Sleep + Cov + APOE4,
    'Sleep_Cov_Brain_APOE': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + APOE4,
    'Sleep_Cov_Brain_APOE_Shuffle': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + APOE4
}

y_dict = {
    'Stroop': 'Stroop_Test',
    'Memory': 'Memory_Test',
    'Stroop_rgo_age': 'Stroop_rgo_age',
    'Memory_rgo_age': 'Memory_rgo_age'
}

X = X_dict[feature_comb]
y = y_dict[target]

# ==========================
# Paths: load from models, save to Results
# ==========================
data_save_path = (
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
)

# Model folder (only for LOADING SHAP / SHAP-IQ)
model_case_path = (
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/'
    f'Out-of-sample_validation/models/AutoGluon/{target}/{feature_comb}/'
)

# New results folder for all plots
results_root = (
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/'
    'Out-of-sample_validation/Results/OOCV_SHAP_clustering/'
)
case_results_path = os.path.join(results_root, target, feature_comb)
os.makedirs(case_results_path, exist_ok=True)

site_plot_dir    = os.path.join(case_results_path, 'site_plots')
cluster_plot_dir = os.path.join(case_results_path, 'cluster_plots')

print("Results will be saved under:", case_results_path)

# ==========================
# Load datasets
# ==========================
df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test', 'Memory_Test']).reset_index(drop=True)

df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed_target_cleaned.csv')
df_Liege = df_Liege.dropna(subset=['Stroop_Test', 'Memory_Test']).reset_index(drop=True)

df_VETSA = pd.read_csv(data_save_path + 'VETSA_dataset_renamed_target_cleaned.csv')
df_VETSA = df_VETSA.dropna(subset=['Stroop_Test', 'Memory_Digit_Test', 'Memory_Letter_Test']).reset_index(drop=True)
df_VETSA['PSG_Sleep_Dur'] = float('nan')
df_VETSA['PSG_Sleep_Eff'] = float('nan')

df_KI = pd.read_csv(data_save_path + 'KI_dataset_renamed_target_cleaned.csv')
df_KI = df_KI.dropna(subset=['Memory_Test']).reset_index(drop=True)

df_Pitts = pd.read_csv(data_save_path + 'Pitts_dataset_renamed_target_cleaned.csv')
df_Pitts = df_Pitts.dropna(subset=['Executive_Functioning_Test', 'Memory_Letter_Test', 'Memory_Spatial_Test']).reset_index(drop=True)
df_Pitts['APOE4'] = None

df_Juelich_sess_1 = pd.read_csv(
    data_save_path + 'Juelich_1_all_renamed_target_cleaned.csv', index_col=0
)
df_Juelich_sess_1 = df_Juelich_sess_1.dropna(
    subset=['Letter_Sensitivity_Test', 'Spatial_Sensitivity_Test']
).reset_index(drop=True)
df_Juelich_sess_1['APOE4'] = None

df_Juelich_sess_2 = pd.read_csv(
    data_save_path + 'Juelich_2_all_renamed_target_cleaned.csv', index_col=0
)
df_Juelich_sess_2 = df_Juelich_sess_2.dropna(
    subset=['Letter_Sensitivity_Test', 'Spatial_Sensitivity_Test']
).reset_index(drop=True)
df_Juelich_sess_2['APOE4'] = None

df_Juelich_SD = pd.read_csv(
    data_save_path + 'Juelich_SD_all_renamed_target_cleaned.csv', index_col=0
)
df_Juelich_SD = df_Juelich_SD.dropna(
    subset=['Letter_Sensitivity_Test', 'Spatial_Sensitivity_Test']
).reset_index(drop=True)
df_Juelich_SD['APOE4'] = None

# ==========================
# Dataset preprocessing
# ==========================
def prep(df, rs=33):
    df = df.copy()

    # unit conversion
    df = utils.convert_units(df, sleep_dur_cols, sleep_eff_cols)

    # brain-size normalisation
    for g in [Thickness_DK, Thickness_Schaefer, Area_DK, Area_Schaefer]:
        df[g] = df[g].div(df[g].sum(axis=1), axis=0)
    df[Subcortical] = df[Subcortical].div(df['EstimatedTotalIntraCranialVol'], axis=0)

    # memory score → %
    if 'Memory_Test' in df.columns:
        df['Memory_Test'] = df['Memory_Test'] / 16 * 100

    # optional shuffles (still respect feature_comb)
    r = dict(frac=1, random_state=rs)
    if feature_comb in ['Sleep_Shuffle_Cov', 'Sleep_Shuffle_Cov_Subcor',
                        'Sleep_Shuffle_Subcor']:
        df[Sleep] = df[Sleep].sample(**r).reset_index(drop=True)
    if feature_comb in ['Sleep_Cov_Subcor_Shuffle', 'Cov_Subcor_Shuffle']:
        df[Subcortical] = df[Subcortical].sample(**r).reset_index(drop=True)
    if feature_comb in ['Cov_Brain_Shuffle', 'Sleep_Cov_Brain_Shuffle']:
        for g in [Thickness_DK, Thickness_Schaefer, Area_DK, Area_Schaefer, Subcortical]:
            df[g] = df[g].sample(**r).reset_index(drop=True)
    if 'APOE_Shuffle' in feature_comb:
        df[APOE4] = df[APOE4].sample(**r).reset_index(drop=True)

    # dtypes
    df[['SEX', 'APOE4']] = df[['SEX', 'APOE4']].astype('category')
    float_cols = (Sleep + ['Age_at_Scan', 'BMI', TIV] +
                  Thickness_DK + Thickness_Schaefer +
                  Area_DK + Area_Schaefer + Subcortical)
    df[float_cols] = df[float_cols].astype('float64')

    return df


df_SHIP_ml           = prep(df_SHIP)
df_Liege_ml          = prep(df_Liege)
df_VETSA_ml          = prep(df_VETSA)
df_KI_ml             = prep(df_KI)
df_Pitts_ml          = prep(df_Pitts)
df_Juelich_sess_1_ml = prep(df_Juelich_sess_1)
df_Juelich_sess_2_ml = prep(df_Juelich_sess_2)
df_Juelich_SD_ml     = prep(df_Juelich_SD)

# ==========================
# Load SHAP & SHAP-IQ (from models folder)
# ==========================
SHAP_path   = model_case_path + 'SHAP/'
shapiq_path = model_case_path + 'shapiq/'

# SHAP-IQ
with open(shapiq_path + 'ivs_SHIP_SHIP.pkl', 'rb') as f:
    ivs_SHIP_SHIP = pickle.load(f)
with open(shapiq_path + 'ivs_SHIP_Liege.pkl', 'rb') as f:
    ivs_SHIP_Liege = pickle.load(f)
with open(shapiq_path + 'ivs_SHIP_VETSA.pkl', 'rb') as f:
    ivs_SHIP_VETSA = pickle.load(f)
with open(shapiq_path + 'ivs_SHIP_Pitts.pkl', 'rb') as f:
    ivs_SHIP_Pitts = pickle.load(f)

if target == 'Memory':
    with open(shapiq_path + 'ivs_SHIP_KI.pkl', 'rb') as f:
        ivs_SHIP_KI = pickle.load(f)
    with open(shapiq_path + 'ivs_SHIP_Juelichsess-1.pkl', 'rb') as f:
        ivs_SHIP_Juelich_sess_1 = pickle.load(f)
    with open(shapiq_path + 'ivs_SHIP_Juelichsess-2.pkl', 'rb') as f:
        ivs_SHIP_Juelich_sess_2 = pickle.load(f)
    with open(shapiq_path + 'ivs_SHIP_Juelichsess-SD.pkl', 'rb') as f:
        ivs_SHIP_Juelich_SD = pickle.load(f)

# SHAP
with open(SHAP_path + 'explanation_train_set_train.pkl', 'rb') as f:
    explanation_train_set_train = pickle.load(f)
with open(SHAP_path + 'explanation_train_set_test_Liege.pkl', 'rb') as f:
    explanation_train_set_test_Liege = pickle.load(f)
with open(SHAP_path + 'explanation_train_set_test_VETSA.pkl', 'rb') as f:
    explanation_train_set_test_VETSA = pickle.load(f)
with open(SHAP_path + 'explanation_train_set_test_Pitts.pkl', 'rb') as f:
    explanation_train_set_test_Pitts = pickle.load(f)

if target == 'Memory':
    with open(SHAP_path + 'explanation_train_set_test_KI.pkl', 'rb') as f:
        explanation_train_set_test_KI = pickle.load(f)
    with open(SHAP_path + 'explanation_train_set_test_Juelich_sess-1.pkl', 'rb') as f:
        explanation_train_set_test_Juelich_sess_1 = pickle.load(f)
    with open(SHAP_path + 'explanation_train_set_test_Juelich_sess-2.pkl', 'rb') as f:
        explanation_train_set_test_Juelich_sess_2 = pickle.load(f)
    with open(SHAP_path + 'explanation_train_set_test_Juelich_sess-SD.pkl', 'rb') as f:
        explanation_train_set_test_Juelich_SD = pickle.load(f)

# ==========================
# Site-level plots
# ==========================
def make_site_plots(
        shap_results: dict,
        shapiq_results: dict,
        dataframes: dict,
        feature_names: list[str],
        save_dir: str | None = None,
        max_display: int | None = 20,
        abbreviate: bool = False,
        show: bool = True,
    ):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    pretty_feats = pretty_names(feature_names)
    sites = shap_results.keys() | shapiq_results.keys() | dataframes.keys()

    for site in sites:
        explanation = shap_results.get(site)
        ivs         = shapiq_results.get(site)
        df_site     = dataframes.get(site)

        if explanation is None or ivs is None or df_site is None:
            print(f"[WARN] Site '{site}' missing SHAP / SHAP-IQ / data; skipped.")
            continue

        # 1) SHAP summary (dot)
        exp = explanation
        exp_pretty = [LABEL_REPLACEMENTS.get(n, n) for n in exp.feature_names]

        shap.summary_plot(
            exp,
            features=exp.data,
            feature_names=exp_pretty,
            plot_type="dot",
            sort=True,
            show=False,
            plot_size=DEFAULT_FIGSIZE_RX,
            max_display=max_display,
        )
        ax = plt.gca()
        ax.set_title(f"{site} – SHAP summary", fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.label.set_size(FS_LABEL)
        ax.yaxis.label.set_size(FS_LABEL)
        plt.tight_layout()

        out1 = os.path.join(save_dir, f"{site}_shap_summary.png") if save_dir else None
        maybe_save_show(out1, show=show)

        # 2) SHAP-IQ bar
        plt.figure(figsize=DEFAULT_FIGSIZE_RX)
        shapiq.plot.bar_plot(
            ivs,
            feature_names=pretty_feats,
            abbreviate=abbreviate,
            max_display=max_display,
            show=False,
        )
        ax = plt.gca()
        ax.set_title(f"{site} – SHAP-IQ bar", fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.label.set_size(FS_LABEL)
        ax.yaxis.label.set_size(FS_LABEL)
        plt.tight_layout()

        out2 = os.path.join(save_dir, f"{site}_shapiq_bar.png") if save_dir else None
        maybe_save_show(out2, show=show)

        # 3) SHAP-IQ beeswarm
        plt.figure(figsize=DEFAULT_FIGSIZE_RX)
        shapiq.plot.beeswarm_plot(
            ivs,
            df_site[feature_names],
            feature_names=pretty_feats,
            abbreviate=abbreviate,
            show=False,
        )
        ax = plt.gca()
        ax.set_title(f"{site} – SHAP-IQ beeswarm", fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.label.set_size(FS_LABEL)
        ax.yaxis.label.set_size(FS_LABEL)
        plt.tight_layout()

        out3 = os.path.join(save_dir, f"{site}_shapiq_beeswarm.png") if save_dir else None
        maybe_save_show(out3, show=show)

        print(f"[{site}] site-level SHAP / SHAP-IQ plots done.")


shap_results = {
    "Greifswald":  explanation_train_set_train,
    "Liège":       explanation_train_set_test_Liege,
    "San Diego":   explanation_train_set_test_VETSA,
    "Pittsburgh":  explanation_train_set_test_Pitts,
}
if target == 'Memory':
    shap_results["Stockholm"]          = explanation_train_set_test_KI
    shap_results["Jülich Session 1"]   = explanation_train_set_test_Juelich_sess_1
    shap_results["Jülich Session 2"]   = explanation_train_set_test_Juelich_sess_2
    shap_results["Jülich Session SD"]  = explanation_train_set_test_Juelich_SD

shapiq_results = {
    "Greifswald":  ivs_SHIP_SHIP,
    "Liège":       ivs_SHIP_Liege,
    "San Diego":   ivs_SHIP_VETSA,
    "Pittsburgh":  ivs_SHIP_Pitts,
}
if target == 'Memory':
    shapiq_results["Stockholm"]         = ivs_SHIP_KI
    shapiq_results["Jülich Session 1"]  = ivs_SHIP_Juelich_sess_1
    shapiq_results["Jülich Session 2"]  = ivs_SHIP_Juelich_sess_2
    shapiq_results["Jülich Session SD"] = ivs_SHIP_Juelich_SD

dataframes = {
    "Greifswald":  df_SHIP_ml,
    "Liège":       df_Liege_ml,
    "San Diego":   df_VETSA_ml,
    "Pittsburgh":  df_Pitts_ml,
}
if target == 'Memory':
    dataframes["Stockholm"]         = df_KI_ml
    dataframes["Jülich Session 1"]  = df_Juelich_sess_1_ml
    dataframes["Jülich Session 2"]  = df_Juelich_sess_2_ml
    dataframes["Jülich Session SD"] = df_Juelich_SD_ml

make_site_plots(
    shap_results,
    shapiq_results,
    dataframes,
    feature_names=X,
    save_dir=site_plot_dir,   # <== all site-level plots saved here
    max_display=20,
    abbreviate=False,
    show=SHOW_PLOTS,
)

# ==========================
# SHIP clustering & rules
# ==========================
def shap_explanation_to_df(
    explanation: "shap.Explanation | list | np.ndarray",
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    if hasattr(explanation, "values") and hasattr(explanation, "feature_names"):
        values = explanation.values
        names  = list(explanation.feature_names)
    elif isinstance(explanation, list) and explanation:
        first = explanation[0]
        if not (hasattr(first, "values") and hasattr(first, "feature_names")):
            raise ValueError("Items in explanation list are not SHAP Explanation objects.")
        values = first.values
        names  = list(first.feature_names)
    elif isinstance(explanation, np.ndarray):
        values = explanation
        if feature_names is None:
            raise ValueError("Must supply feature_names when explanation is a NumPy array.")
        names = feature_names
    else:
        raise TypeError("Unrecognized explanation type.")

    if values.shape[1] != len(names):
        raise ValueError(
            f"SHAP matrix has {values.shape[1]} columns but {len(names)} feature names."
        )

    return pd.DataFrame(values, columns=names)


SHIP_shap_df = shap_explanation_to_df(explanation_train_set_train)

SHIP_shap_embed = SHIP_shap_df.values

if 'Stroop' in target:
    best_k = 4
elif 'Memory' in target:
    best_k = 6
else:
    best_k = 4  # default fallback

from itertools import cycle, islice


def cluster_palette(labels):
    uniq = np.unique(labels)
    cols = list(islice(cycle(BASE_COLORS), len(uniq)))
    return dict(zip(uniq, cols))


k_final = best_k
spec = SpectralClustering(
    n_clusters=k_final,
    affinity="nearest_neighbors",
    n_neighbors=15,
    assign_labels="kmeans",
    random_state=42
)
labels_final = spec.fit(SHIP_shap_embed).labels_.astype(str)

# df_SHIP_X = df_SHIP_ml[X]
# y_clust   = pd.Series(labels_final, name="cluster").astype(str)
def make_rule_input(df, cols):
    """
    Convert rule input to numeric values while preserving original numeric coding.
    Important: do not use .cat.codes, because SEX 1/2 may become 0/1.
    """
    X_rule = df[cols].copy()

    for c in X_rule.columns:
        if str(X_rule[c].dtype) == "category":
            X_rule[c] = pd.to_numeric(X_rule[c].astype(str), errors="coerce")
        elif X_rule[c].dtype == "object":
            X_rule[c] = pd.to_numeric(X_rule[c], errors="coerce")

    return X_rule


df_SHIP_X = make_rule_input(df_SHIP_ml, X)
feature_cols = list(df_SHIP_X.columns)
y_clust = pd.Series(labels_final, name="cluster").astype(str)

print("SHIP cluster counts:", y_clust.value_counts().to_dict())

from collections import OrderedDict


def fit_skope_by_cluster(
        df_X: pd.DataFrame,
        y_clust: pd.Series,
        *,
        n_estimators: int       = 30,
        recall_min:   float     = 0.30,
        max_depth:    int       = 4,
        max_depth_duplication: int = 6,
        max_samples:  float     = 0.80,
        random_state: int       = 42
) -> OrderedDict[str, SkopeRules]:
    models = OrderedDict()
    for cl in sorted(y_clust.unique(), key=str):
        y_bin = (y_clust == cl).astype(int).values
        sr = SkopeRules(
            feature_names=df_X.columns.tolist(),
            n_estimators=n_estimators,
            recall_min=recall_min,
            max_depth=max_depth,
            max_depth_duplication=max_depth_duplication,
            max_samples=max_samples,
            max_features=None,
            random_state=random_state
        ).fit(df_X, y_bin)
        models[cl] = sr
    return models


sr_models = fit_skope_by_cluster(df_SHIP_X, y_clust)

# %%
# ==========================
# Export subgroup artifacts without SHIP raw data
# ==========================
subgroup_export_dir = os.path.join(case_results_path, "subgroup_artifacts")
os.makedirs(subgroup_export_dir, exist_ok=True)

cluster_rules = {}
rule_info = {}

for cl, sr in sr_models.items():
    cl = str(cl)

    if not sr.rules_:
        print(f"[WARN] Cluster {cl}: no rule found.")
        continue

    # only export the top rule per cluster
    rule_str, (prec, rec, _) = sr.rules_[0]

    cluster_rules[cl] = [rule_str]
    rule_info[cl] = {
        "rule": rule_str,
        "precision": float(prec),
        "recall": float(rec),
    }

ship_cluster_counts = {
    str(k): int(v)
    for k, v in y_clust.value_counts().to_dict().items()
}

subgroup_artifacts = {
    "target": target,
    "feature_comb": feature_comb,

    # Required for applying the rules to EMC
    "feature_cols": feature_cols,
    "cluster_rules": cluster_rules,

    # Useful metadata, no subject-level data
    "rule_info": rule_info,
    "ship_cluster_counts": ship_cluster_counts,
}

subgroup_artifact_path = os.path.join(
    subgroup_export_dir,
    f"subgroup_artifacts_{target}_{feature_comb}.pkl"
)

with open(subgroup_artifact_path, "wb") as f:
    pickle.dump(subgroup_artifacts, f)

print(f"Saved subgroup artifacts to: {subgroup_artifact_path}")

rule_report_path = os.path.join(
    subgroup_export_dir,
    f"subgroup_rules_{target}_{feature_comb}.txt"
)

with open(rule_report_path, "w") as f:
    f.write(f"Target: {target}\n")
    f.write(f"Feature combination: {feature_comb}\n")
    f.write(f"Number of clusters: {len(cluster_rules)}\n\n")

    f.write("SHIP cluster counts:\n")
    for cl, n in ship_cluster_counts.items():
        f.write(f"  Cluster {cl}: n={n}\n")

    f.write("\nRules:\n")
    for cl, info in rule_info.items():
        f.write(f"\nCluster {cl}\n")
        f.write(f"Rule: {info['rule']}\n")
        f.write(f"Precision: {info['precision']:.4f}\n")
        f.write(f"Recall: {info['recall']:.4f}\n")

print(f"Saved subgroup rule report to: {rule_report_path}")

# %%
# for cl, sr in sr_models.items():
#     print(f"\n=== Cluster {cl} ===")
#     if not sr.rules_:
#         print("No rule met precision/recall thresholds.")
#         continue
#     rule_str, (prec, rec, _) = sr.rules_[0]
#     pretty_rule = " AND ".join(rule_str.split(" and "))
#     print(f"IF {pretty_rule}")
#     print(f"→ precision = {prec:.3f}, recall (coverage) = {rec:.3f}")
#
# # ==========================
# # Apply rules to other sites
# # ==========================
# def process_site(df_site, site_name, *, rules, cols, unknown='unassigned'):
#     X_site = df_site[cols].copy()
#     for c in X_site.select_dtypes('category'):
#         X_site[c] = X_site[c].cat.codes.astype(float)
#
#     def _label(row):
#         for cl, rl in rules.items():
#             for r in rl:
#                 if row.to_frame().T.query(r).shape[0]:
#                     return cl
#         return unknown
#
#     labels = X_site.apply(_label, axis=1)
#
#     print(f"\n{site_name}:")
#     print(labels.value_counts(dropna=False))
#
#     return labels
#
#
# cluster_rules = {cl: [sr.rules_[0][0]] for cl, sr in sr_models.items()}
# feature_cols  = X
#
# labels_Liege = process_site(
#     df_Liege_ml, "Liège",
#     rules=cluster_rules,
#     cols=feature_cols
# )
#
# labels_VETSA = process_site(
#     df_VETSA_ml, "San Diego",
#     rules=cluster_rules,
#     cols=feature_cols
# )
#
# labels_Pitts = process_site(
#     df_Pitts_ml, "Pittsburgh",
#     rules=cluster_rules,
#     cols=feature_cols
# )
#
# if target == 'Memory':
#     labels_KI = process_site(
#         df_KI_ml, "Stockholm",
#         rules=cluster_rules,
#         cols=feature_cols
#     )
#     labels_Juelich_sess_1 = process_site(
#         df_Juelich_sess_1_ml, "Jülich Session 1",
#         rules=cluster_rules,
#         cols=feature_cols
#     )
#     labels_Juelich_sess_2 = process_site(
#         df_Juelich_sess_2_ml, "Jülich Session 2",
#         rules=cluster_rules,
#         cols=feature_cols
#     )
#     labels_Juelich_SD = process_site(
#         df_Juelich_SD_ml, "Jülich Session SD",
#         rules=cluster_rules,
#         cols=feature_cols
#     )
#
# # ==========================
# # Cluster-wise plots per site
# # ==========================
# def plot_si_graph(
#     iv_obj,
#     feature_names: list[str],
#     title: str,
#     fig_size=DEFAULT_FIGSIZE_SQ,
#     out: str | None = None,
#     show: bool = True,
#     size_factor: float = 1.0,
#     min_max_interactions: tuple | None = None,
# ):
#     pretty_feats = pretty_names(feature_names)
#     fig, ax = shapiq.si_graph_plot(
#         interaction_values=iv_obj,
#         feature_names=pretty_feats,
#         show=False,
#         min_max_order=(1, 2),
#         size_factor=size_factor,
#         min_max_interactions=min_max_interactions,
#     )
#     if fig_size is not None:
#         fig.set_size_inches(fig_size[0], fig_size[1], forward=True)
#
#     for txt in ax.texts:
#         txt.set_fontsize(10)
#
#     ax.set_title(title, fontsize=FS_TITLE-2, pad=20)
#     fig.tight_layout()
#
#     if out:
#         p = Path(out)
#         p.parent.mkdir(parents=True, exist_ok=True)
#         suf = p.suffix.lower()
#
#         if suf == ".png":
#             fig.savefig(p, dpi=300, bbox_inches="tight")
#             fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
#         elif suf == ".svg":
#             fig.savefig(p, bbox_inches="tight")
#         elif suf == "":
#             fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
#             fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
#         else:
#             fig.savefig(p, dpi=300, bbox_inches="tight")
#
#
# def make_cluster_plots(
#         *,
#         labels:        np.ndarray | pd.Series,
#         explanation:   "shap.Explanation",
#         ivs:           list,
#         feature_names: list[str],
#         site_name:     str,
#         save_dir:      str | None = None,
#         min_max_order: tuple[int, int] = (1, 2),
#         show:          bool = True,
#         size_factor:   float = 1.0,
#         max_display:   int = 20,
# ):
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#
#     labels = np.asarray(labels)
#     unique_labels = np.unique(labels)
#     pretty_feats = pretty_names(feature_names)
#
#     for cl in unique_labels:
#         idx = np.where(labels == cl)[0]
#         if idx.size == 0:
#             continue
#
#         # 1) Mean interaction network
#         iv_avg = aggregate_interaction_values([ivs[i] for i in idx],
#                                               aggregation="mean")
#
#         net_out = (
#             os.path.join(save_dir, f"{site_name}_cl{cl}_network.png")
#             if save_dir else None
#         )
#         plot_si_graph(
#             iv_avg,
#             feature_names,
#             title=f"{site_name} – Cluster {cl} (n={len(idx)})",
#             fig_size=DEFAULT_FIGSIZE_SQ,
#             out=net_out,
#             show=show,
#             size_factor=size_factor,
#             min_max_interactions=None,
#         )
#
#         # 2) SHAP summary (dot)
#         exp_cl = explanation[idx]
#         exp_pretty = [LABEL_REPLACEMENTS.get(n, n)
#                       for n in exp_cl.feature_names]
#
#         shap.summary_plot(
#             exp_cl,
#             features=exp_cl.data,
#             feature_names=exp_pretty,
#             plot_type="dot",
#             sort=True,
#             show=False,
#             plot_size=DEFAULT_FIGSIZE_RX,
#             max_display=max_display,
#         )
#         ax = plt.gca()
#         ax.set_title(f"{site_name} – Cluster {cl} (n={len(idx)})", fontsize=FS_TITLE)
#         ax.tick_params(labelsize=FS_TICK)
#         ax.xaxis.label.set_size(FS_LABEL)
#         ax.yaxis.label.set_size(FS_LABEL)
#         plt.tight_layout()
#
#         swarm_out = (
#             os.path.join(save_dir, f"{site_name}_cl{cl}_shap_summary.png")
#             if save_dir else None
#         )
#         maybe_save_show(swarm_out, show=show)
#
#         # 3) SHAP-IQ bar plot
#         plt.figure(figsize=DEFAULT_FIGSIZE_RX)
#         shapiq.plot.bar_plot(
#             [ivs[i] for i in idx],
#             feature_names=pretty_feats,
#             max_display=max_display,
#             abbreviate=False,
#             show=False,
#         )
#         ax = plt.gca()
#         ax.set_title(f"{site_name} – Cluster {cl} (n={len(idx)})", fontsize=FS_TITLE)
#         ax.tick_params(labelsize=FS_TICK)
#         ax.xaxis.label.set_size(FS_LABEL)
#         ax.yaxis.label.set_size(FS_LABEL)
#         plt.tight_layout()
#
#         bar_out = (
#             os.path.join(save_dir, f"{site_name}_cl{cl}_shapiq_bar.png")
#             if save_dir else None
#         )
#         maybe_save_show(bar_out, show=show)
#
#         print(f"[{site_name}] Cluster {cl} plots done.")
#
#
# # Cluster plots for each external site – all saved under cluster_plot_dir
# make_cluster_plots(
#     labels        = labels_Liege.values,
#     explanation   = explanation_train_set_test_Liege,
#     ivs           = ivs_SHIP_Liege,
#     feature_names = feature_cols,
#     site_name     = "Liège",
#     save_dir      = cluster_plot_dir,
#     show          = SHOW_PLOTS,
# )
#
# make_cluster_plots(
#     labels        = labels_VETSA.values,
#     explanation   = explanation_train_set_test_VETSA,
#     ivs           = ivs_SHIP_VETSA,
#     feature_names = feature_cols,
#     site_name     = "San Diego",
#     save_dir      = cluster_plot_dir,
#     show          = SHOW_PLOTS,
# )
#
# make_cluster_plots(
#     labels        = labels_Pitts.values,
#     explanation   = explanation_train_set_test_Pitts,
#     ivs           = ivs_SHIP_Pitts,
#     feature_names = feature_cols,
#     site_name     = "Pittsburgh",
#     save_dir      = cluster_plot_dir,
#     show          = SHOW_PLOTS,
# )
#
# if target == 'Memory':
#     make_cluster_plots(
#         labels        = labels_KI.values,
#         explanation   = explanation_train_set_test_KI,
#         ivs           = ivs_SHIP_KI,
#         feature_names = feature_cols,
#         site_name     = "Stockholm",
#         save_dir      = cluster_plot_dir,
#         show          = SHOW_PLOTS,
#     )
#     make_cluster_plots(
#         labels        = labels_Juelich_sess_1.values,
#         explanation   = explanation_train_set_test_Juelich_sess_1,
#         ivs           = ivs_SHIP_Juelich_sess_1,
#         feature_names = feature_cols,
#         site_name     = "Jülich Session 1",
#         save_dir      = cluster_plot_dir,
#         show          = SHOW_PLOTS,
#     )
#     make_cluster_plots(
#         labels        = labels_Juelich_sess_2.values,
#         explanation   = explanation_train_set_test_Juelich_sess_2,
#         ivs           = ivs_SHIP_Juelich_sess_2,
#         feature_names = feature_cols,
#         site_name     = "Jülich Session 2",
#         save_dir      = cluster_plot_dir,
#         show          = SHOW_PLOTS,
#     )
#     make_cluster_plots(
#         labels        = labels_Juelich_SD.values,
#         explanation   = explanation_train_set_test_Juelich_SD,
#         ivs           = ivs_SHIP_Juelich_SD,
#         feature_names = feature_cols,
#         site_name     = "Jülich Session SD",
#         save_dir      = cluster_plot_dir,
#         show          = SHOW_PLOTS,
#     )
#
# print("\nDone. All figures saved under:", case_results_path)
