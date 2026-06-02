import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set_context("paper")
import shap
import shapiq
from shapiq.interaction_values import aggregate_interaction_values
import pickle

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array

import collections, collections.abc, six, sklearn
collections.Iterable = collections.abc.Iterable
sklearn.externals.six = six
from skrules import SkopeRules
from cluster_explorer import Explainer as CE_explainer

import umap.umap_ as umap

from joblib import dump, load

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib as mpl
# Tell Matplotlib: “Whenever I ask for sans-serif, try Arial first”
mpl.rcParams['font.family']      = 'sans-serif'
mpl.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='SHAP-IQ for AutoGluon, SHIP_Trend dataset.')
parser.add_argument('feature_comb', type=str, help='''Name of the feature combination to be used.
    Available combinations are:
    - Sleep: Uses sleep features only.
    - Cov: Uses covariates only.
    - Sleep_Cov: Uses sleep features combined with covariates.
    - Sleep_Shuffle_Cov: Uses sleep features (shuffled) and covariates.
    - Brain: Uses brain features only.
    - Sleep_Cov_Brain: Uses sleep features, covariates, and brain features.
    - Sleep_Cov_Brain_Shuffle: Uses sleep features, covariates, and brain features (shuffled).
    - Sleep_APOE: Uses sleep features and APOE4.
    - Sleep_APOE_Shuffle: Uses sleep features (shuffled) and APOE4.
    - Sleep_Cov_APOE: Uses sleep features, covariates, and APOE4.
    - Sleep_Cov_APOE_Shuffle: Uses sleep features, covariates (shuffled), and APOE4.
    - Sleep_Cov_Brain_APOE: Uses sleep features, covariates, brain features, and APOE4.
    - Sleep_Cov_Brain_APOE_Shuffle: Uses sleep features, covariates, brain features (shuffled), and APOE4.
''')

parser.add_argument('target', type=str, help='''Name of the target to be predicted.
    Available targets are:
    - Stroop: Stroop_Test
    - Memory: Memory_Test
    - Stroop_rgo_age: Stroop_rgo_age
    - Memory_rgo_age: Memory_rgo_age
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target

print(f"\nStarting SHAP-IQ for AutoGluon pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
"""
Define feature lists
"""
# load the feature lists from feature_lists.pkl
feature_lists_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/feature_lists.pkl'  # where the feature lists are saved
with open(feature_lists_path, 'rb') as f:
    feature_lists = pickle.load(f)

Sleep = feature_lists[0]
Cov = feature_lists[1]
APOE4 = feature_lists[2]
TIV = feature_lists[3]
Thickness_DK = feature_lists[4]
Thickness_Schaefer = feature_lists[5]
Area_DK = feature_lists[6]
Area_Schaefer = feature_lists[7]
Subcortical = feature_lists[8]
targets = feature_lists[9]

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

# %%
"""
Load datasets
"""
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/{target}/{feature_comb}/'

df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test', 'Memory_Test'])
df_SHIP = df_SHIP.reset_index(drop=True)

df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed_target_cleaned.csv')
df_Liege = df_Liege.dropna(subset=['Stroop_Test', 'Memory_Test'])
df_Liege = df_Liege.reset_index(drop=True)

df_VETSA = pd.read_csv(data_save_path + 'VETSA_dataset_renamed_target_cleaned.csv')
df_VETSA = df_VETSA.dropna(subset=['Stroop_Test', 'Memory_Digit_Test', 'Memory_Letter_Test'])
df_VETSA = df_VETSA.reset_index(drop=True)
df_VETSA['PSG_Sleep_Dur'] = float('nan')
df_VETSA['PSG_Sleep_Eff'] = float('nan')

df_KI = pd.read_csv(data_save_path + 'KI_dataset_renamed_target_cleaned.csv')
df_KI = df_KI.dropna(subset=['Memory_Test'])
df_KI = df_KI.reset_index(drop=True)

df_Pitts = pd.read_csv(data_save_path + 'Pitts_dataset_renamed_target_cleaned.csv')
df_Pitts = df_Pitts.dropna(subset=['Executive_Functioning_Test', 'Memory_Letter_Test', 'Memory_Spatial_Test'])
df_Pitts = df_Pitts.reset_index(drop=True)
df_Pitts['APOE4'] = None

df_Juelich_sess_1 = pd.read_csv(data_save_path + 'Juelich_1_all_renamed_target_cleaned.csv', index_col=0)
df_Juelich_sess_1 = df_Juelich_sess_1.dropna(subset=['Letter_Sensitivity_Test', 'Spatial_Sensitivity_Test'])
df_Juelich_sess_1 = df_Juelich_sess_1.reset_index(drop=True)
df_Juelich_sess_1['APOE4'] = None
df_Juelich_sess_2 = pd.read_csv(data_save_path + 'Juelich_2_all_renamed_target_cleaned.csv', index_col=0)
df_Juelich_sess_2 = df_Juelich_sess_2.dropna(subset=['Letter_Sensitivity_Test', 'Spatial_Sensitivity_Test'])
df_Juelich_sess_2 = df_Juelich_sess_2.reset_index(drop=True)
df_Juelich_sess_2['APOE4'] = None
df_Juelich_SD = pd.read_csv(data_save_path + 'Juelich_SD_all_renamed_target_cleaned.csv', index_col=0)
df_Juelich_SD = df_Juelich_SD.dropna(subset=['Letter_Sensitivity_Test', 'Spatial_Sensitivity_Test'])
df_Juelich_SD = df_Juelich_SD.reset_index(drop=True)
df_Juelich_SD['APOE4'] = None

# %%
"""
Dataset preprocessing
"""
def prep(df, rs=33):
    """Fast one-liner replacement for the long preprocessing block."""
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

    # optional shuffles
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

# Preprocess datasets
df_SHIP_ml = prep(df_SHIP)
df_Liege_ml = prep(df_Liege)
df_VETSA_ml = prep(df_VETSA)
df_KI_ml = prep(df_KI)
df_Pitts_ml = prep(df_Pitts)
df_Juelich_sess_1_ml = prep(df_Juelich_sess_1)
df_Juelich_sess_2_ml = prep(df_Juelich_sess_2)
df_Juelich_SD_ml = prep(df_Juelich_SD)

# %%
"""
Load SHAP and SHAP-IQ results for different datasets
"""
case_results_path = trained_models_path
SHAP_path = case_results_path + 'SHAP/'
shapiq_path = case_results_path + 'shapiq/'

# load saved shapiq results by pickle
ivs_saved_path_SHIP_SHIP = shapiq_path + 'ivs_SHIP_SHIP.pkl'
with open(ivs_saved_path_SHIP_SHIP, 'rb') as f:
    ivs_SHIP_SHIP = pickle.load(f)

ivs_saved_path_SHIP_Liege = shapiq_path + 'ivs_SHIP_Liege.pkl'
with open(ivs_saved_path_SHIP_Liege, 'rb') as f:
    ivs_SHIP_Liege = pickle.load(f)

ivs_saved_path_SHIP_VETSA = shapiq_path + 'ivs_SHIP_VETSA.pkl'
with open(ivs_saved_path_SHIP_VETSA, 'rb') as f:
    ivs_SHIP_VETSA = pickle.load(f)

ivs_saved_path_SHIP_Pitts = shapiq_path + 'ivs_SHIP_Pitts.pkl'
with open(ivs_saved_path_SHIP_Pitts, 'rb') as f:
    ivs_SHIP_Pitts = pickle.load(f)

if target == 'Memory':
    ivs_saved_path_SHIP_KI = shapiq_path + 'ivs_SHIP_KI.pkl'
    with open(ivs_saved_path_SHIP_KI, 'rb') as f:
        ivs_SHIP_KI = pickle.load(f)

    ivs_saved_path_SHIP_Juelich_sess_1= shapiq_path + 'ivs_SHIP_Juelichsess-1.pkl'
    with open(ivs_saved_path_SHIP_Juelich_sess_1, 'rb') as f:
        ivs_SHIP_Juelich_sess_1 = pickle.load(f)

    ivs_saved_path_SHIP_Juelich_sess_2 = shapiq_path + 'ivs_SHIP_Juelichsess-2.pkl'
    with open(ivs_saved_path_SHIP_Juelich_sess_2, 'rb') as f:
        ivs_SHIP_Juelich_sess_2 = pickle.load(f)

    ivs_saved_path_SHIP_Juelich_SD = shapiq_path + 'ivs_SHIP_Juelichsess-SD.pkl'
    with open(ivs_saved_path_SHIP_Juelich_SD, 'rb') as f:
        ivs_SHIP_Juelich_SD = pickle.load(f)

# load saved SHAP results by pickle
shap_saved_path_SHIP_SHIP = SHAP_path + 'explanation_train_set_train.pkl'
with open(shap_saved_path_SHIP_SHIP, 'rb') as f:
    explanation_train_set_train = pickle.load(f)

shap_saved_path_SHIP_Liege = SHAP_path + 'explanation_train_set_test_Liege.pkl'
with open(shap_saved_path_SHIP_Liege, 'rb') as f:
    explanation_train_set_test_Liege = pickle.load(f)

shap_saved_path_SHIP_VETSA = SHAP_path + 'explanation_train_set_test_VETSA.pkl'
with open(shap_saved_path_SHIP_VETSA, 'rb') as f:
    explanation_train_set_test_VETSA = pickle.load(f)

shap_saved_path_SHIP_Pitts = SHAP_path + 'explanation_train_set_test_Pitts.pkl'
with open(shap_saved_path_SHIP_Pitts, 'rb') as f:
    explanation_train_set_test_Pitts = pickle.load(f)

if target == 'Memory':
    shap_saved_path_SHIP_KI = SHAP_path + 'explanation_train_set_test_KI.pkl'
    with open(shap_saved_path_SHIP_KI, 'rb') as f:
        explanation_train_set_test_KI = pickle.load(f)

    shap_saved_path_SHIP_Juelich_sess_1 = SHAP_path + 'explanation_train_set_test_Juelich_sess-1.pkl'
    with open(shap_saved_path_SHIP_Juelich_sess_1, 'rb') as f:
        explanation_train_set_test_Juelich_sess_1 = pickle.load(f)

    shap_saved_path_SHIP_Juelich_sess_2 = SHAP_path + 'explanation_train_set_test_Juelich_sess-2.pkl'
    with open(shap_saved_path_SHIP_Juelich_sess_2, 'rb') as f:
        explanation_train_set_test_Juelich_sess_2 = pickle.load(f)

    shap_saved_path_SHIP_Juelich_SD = SHAP_path + 'explanation_train_set_test_Juelich_sess-SD.pkl'
    with open(shap_saved_path_SHIP_Juelich_SD, 'rb') as f:
        explanation_train_set_test_Juelich_SD = pickle.load(f)

# %%
def make_site_plots(
        shap_results: dict,
        shapiq_results: dict,
        dataframes: dict,
        feature_names: list,
        save_dir: str | None = None,
        max_display: int | None = None,
        abbreviate: bool = False,
        show: bool = True,
    ):
    """
    Generate a SHAP beeswarm, SHAPiQ bar plot, and SHAPiQ beeswarm for each site.

    Parameters
    ----------
    shap_results      dict[str, shap._explanation.Explanation]
        Mapping site → SHAP Explanation object.
    shapiq_results    dict[str, shapiq.interaction_values.InteractionValues]
        Mapping site → SHAPiQ interaction-value object (ivs).
    dataframes        dict[str, pandas.DataFrame]
        Original feature‐matrix per site; needed for the SHAPiQ beeswarm.
    feature_names     list[str]
        Columns to display (order must match model input).
    save_dir          str | None
        Folder to write PNGs (created if it doesn’t exist).  If None, nothing is saved.
    max_display       int
        How many top features to show.
    abbreviate        bool
        Whether SHAPiQ should abbreviate long feature names.
    show              bool
        Whether to call ``plt.show()`` so plots pop up in notebooks.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # use sites common to both dictionaries; warn if one is missing
    sites = shap_results.keys() | shapiq_results.keys()
    for site in sites:
        explanation = shap_results.get(site)
        ivs         = shapiq_results.get(site)
        df_site     = dataframes.get(site)

        if explanation is None or ivs is None or df_site is None:
            print(f"[WARN] Site '{site}' missing something (shap, ivs, or dataframe); skipped.")
            continue

        ####################################################################
        # 1) SHAP beeswarm
        ####################################################################
        plt.figure()
        shap.plots.beeswarm(
            explanation,
            clustering=False,
            max_display=max_display,
            show=False,
        )
        plt.title(f"{site} – SHAP beeswarm")
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{site}_shap_beeswarm.png"), dpi=300)
        if show:
            plt.show()
        plt.close()

        ####################################################################
        # 2) SHAPiQ bar plot (global importances)
        ####################################################################
        plt.figure()
        shapiq.plot.bar_plot(
            ivs,
            feature_names=feature_names,
            abbreviate=abbreviate,
            max_display=max_display,
            show=False,
        )
        plt.title(f"{site} – SHAP-IQ bar plot")
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{site}_shapiq_bar.png"), dpi=300)
        if show:
            plt.show()
        plt.close()

        ####################################################################
        # 3) SHAPiQ beeswarm (per-sample contributions)
        ####################################################################
        plt.figure()
        shapiq.plot.beeswarm_plot(
            ivs,
            df_site[feature_names],
            feature_names=feature_names,
            abbreviate=abbreviate,
            show=False,
        )
        plt.title(f"{site} – SHAP-IQ beeswarm")
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{site}_shapiq_beeswarm.png"), dpi=300)
        if show:
            plt.show()
        plt.close()

# %%
shap_results = {
    "Greifswald": explanation_train_set_train,
    "Liège": explanation_train_set_test_Liege,
    "San Diego": explanation_train_set_test_VETSA,
    "Pittsburgh": explanation_train_set_test_Pitts,
}
if target == 'Memory':
    shap_results["Stockholm"] = explanation_train_set_test_KI
    shap_results["Jülich Session 1"] = explanation_train_set_test_Juelich_sess_1
    shap_results["Jülich Session 2"] = explanation_train_set_test_Juelich_sess_2
    shap_results["Jülich Session SD"] = explanation_train_set_test_Juelich_SD

shapiq_results = {
    "Greifswald": ivs_SHIP_SHIP,
    "Liège": ivs_SHIP_Liege,
    "San Diego": ivs_SHIP_VETSA,
    "Pittsburgh": ivs_SHIP_Pitts,
}
if target == 'Memory':
    shapiq_results["Stockholm"] = ivs_SHIP_KI
    shapiq_results["Jülich Session 1"] = ivs_SHIP_Juelich_sess_1
    shapiq_results["Jülich Session 2"] = ivs_SHIP_Juelich_sess_2
    shapiq_results["Jülich Session SD"] = ivs_SHIP_Juelich_SD

dataframes = {
    "Greifswald": df_SHIP_ml,
    "Liège": df_Liege_ml,
    "San Diego": df_VETSA_ml,
    "Pittsburgh": df_Pitts_ml,
}
if target == 'Memory':
    dataframes["Stockholm"] = df_KI_ml
    dataframes["Jülich Session 1"] = df_Juelich_sess_1_ml
    dataframes["Jülich Session 2"] = df_Juelich_sess_2_ml
    dataframes["Jülich Session SD"] = df_Juelich_SD_ml

# %%
# -------------------------------
# Run the helper
# -------------------------------
make_site_plots(
    shap_results,
    shapiq_results,
    dataframes,
    feature_names=X,          # the list/Index you already have
    # save_dir="./plots",       # or None to skip saving
    max_display=20,
    abbreviate=False,
    show=True,                # False if running headless
)

# %%
"""
Get clustering rules from Greifswald dataset
"""
def shap_explanation_to_df(
    explanation: "shap.Explanation | list | np.ndarray",
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert one SHAP result (for one site) to a DataFrame.

    Parameters
    ----------
    explanation   shap.Explanation | list | np.ndarray
        The SHAP object you loaded (can be the usual Explanation,
        a list of Explanations for multi-output, or a raw NumPy array).
    feature_names list[str] | None
        Provide this only if *explanation* is a raw NumPy array or if you
        want to override the names inside the Explanation.

    Returns
    -------
    pd.DataFrame
        Rows = samples, columns = features.
    """
    # ── Standard Explanation ────────────────────────────────────────────────
    if hasattr(explanation, "values") and hasattr(explanation, "feature_names"):
        values = explanation.values
        names  = list(explanation.feature_names)

    # ── Multi-output: use first output -- adjust if needed ──────────────────
    elif isinstance(explanation, list) and explanation:
        first = explanation[0]
        if not (hasattr(first, "values") and hasattr(first, "feature_names")):
            raise ValueError("Items in explanation list are not SHAP Explanation objects.")
        values = first.values
        names  = list(first.feature_names)

    # ── Raw NumPy matrix ────────────────────────────────────────────────────
    elif isinstance(explanation, np.ndarray):
        values = explanation
        if feature_names is None:
            raise ValueError("Must supply feature_names when explanation is a NumPy array.")
        names = feature_names

    else:
        raise TypeError("Unrecognized explanation type.")

    # Final safety check
    if values.shape[1] != len(names):
        raise ValueError(
            f"SHAP matrix has {values.shape[1]} columns but {len(names)} feature names."
        )

    return pd.DataFrame(values, columns=names)

SHIP_shap_df = shap_explanation_to_df(explanation_train_set_train)
Liege_shap_df = shap_explanation_to_df(explanation_train_set_test_Liege)
VETSA_shap_df = shap_explanation_to_df(explanation_train_set_test_VETSA)
Pitts_shap_df = shap_explanation_to_df(explanation_train_set_test_Pitts)
if target == 'Memory':
    KI_shap_df = shap_explanation_to_df(explanation_train_set_test_KI)
    Juelich_sess_1_shap_df = shap_explanation_to_df(explanation_train_set_test_Juelich_sess_1)
    Juelich_sess_2_shap_df = shap_explanation_to_df(explanation_train_set_test_Juelich_sess_2)
    Juelich_SD_shap_df = shap_explanation_to_df(explanation_train_set_test_Juelich_SD)

# %%
SHIP_shap_embed = SHIP_shap_df.values

best_k = None
if 'Stroop' in target:
    best_k = 4
elif 'Memory' in target:
    best_k = 6

from itertools import cycle, islice

BASE_COLORS = [
    "#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
    "#984ea3", "#999999", "#e41a1c", "#dede00"
]

def cluster_palette(labels):
    """
    Return a dict {label: hex‑color} using the fixed colour cycle.
    Works for numeric or string labels.
    """
    uniq = np.unique(labels)  # preserves sort order
    cols = list(islice(cycle(BASE_COLORS), len(uniq)))
    return dict(zip(uniq, cols))  # keep original label types

sns.set_theme(style="ticks", context="paper")

k_final = best_k
spec = SpectralClustering(
    n_clusters=k_final,
    affinity="nearest_neighbors",
    n_neighbors=15,
    assign_labels="kmeans",
    random_state=42
)
labels_final = spec.fit(SHIP_shap_embed).labels_.astype(str)

df_SHIP_X = df_SHIP_ml[X]
y_clust = pd.Series(labels_final, name="cluster").astype(str)

print(df_SHIP_X.shape, y_clust.value_counts().to_dict())

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
    """
    Train one SkopeRules model per unique cluster label.
    Returns an OrderedDict {cluster_label: trained_model}.
    """
    models = OrderedDict()
    for cl in sorted(y_clust.unique(), key=str):
        y_bin = (y_clust == cl).astype(int).values        # 1 = in cluster cl
        sr = SkopeRules(
            feature_names=df_X.columns.tolist(),
            n_estimators=n_estimators,
            recall_min=recall_min,
            max_depth=max_depth,
            max_depth_duplication=max_depth_duplication,
            max_samples=max_samples,
            max_features=None,        # try all features
            random_state=random_state
        ).fit(df_X, y_bin)
        models[cl] = sr
    return models

sr_models = fit_skope_by_cluster(df_SHIP_X, y_clust)

for cl, sr in sr_models.items():
    print(f"\n=== Cluster {cl} ===")
    if not sr.rules_:
        print("No rule met precision/recall thresholds.")
        continue

    rule_str, (prec, rec, _) = sr.rules_[0]   # “_” = count, usually not needed
    # make the rule a little more readable
    pretty = " AND ".join(rule_str.split(" and "))
    print(f"IF {pretty}")
    print(f"→ precision = {prec:.3f}, recall (coverage) = {rec:.3f}")

# %%
"""
apply rules to other datasets SHAP dataframes
"""
def process_site(df_site, site_name, *, rules, cols, unknown='unassigned'):
    """
    Apply SHIP-trained cluster rules to ONE site.

    Parameters
    ----------
    df_site   : pandas.DataFrame   – pre-processed feature table for this site
    site_name : str                – just for printing
    rules     : dict {cluster: [rule1, rule2, …]}
    cols      : list[str]          – column order used during training
    unknown   : str                – label if no rule fires

    Returns
    -------
    pandas.Series  (index = df_site.index) with assigned labels
    """

    # pick & cast the right columns
    X = df_site[cols].copy()
    for c in X.select_dtypes('category'):
        X[c] = X[c].cat.codes.astype(float)

    def _label(row):
        for cl, rl in rules.items():        # first rule that matches wins
            for r in rl:
                if row.to_frame().T.query(r).shape[0]:
                    return cl
        return unknown

    labels = X.apply(_label, axis=1)

    # quick summary
    print(f"\n{site_name}:")
    print(labels.value_counts(dropna=False))

    return labels

cluster_rules = {cl: [sr.rules_[0][0]] for cl, sr in sr_models.items()}
feature_cols = X

labels_Liege = process_site(
    df_Liege_ml, "Liège",
    rules=cluster_rules,
    cols=feature_cols
)

labels_VETSA = process_site(
    df_VETSA_ml, "San Diego",
    rules=cluster_rules,
    cols=feature_cols
)

labels_Pitts = process_site(
    df_Pitts_ml, "Pittsburgh",
    rules=cluster_rules,
    cols=feature_cols
)

if target == 'Memory':
    labels_KI = process_site(
        df_KI_ml, "Stockholm",
        rules=cluster_rules,
        cols=feature_cols
    )

    labels_Juelich_sess_1 = process_site(
        df_Juelich_sess_1_ml, "Jülich Session 1",
        rules=cluster_rules,
        cols=feature_cols
    )

    labels_Juelich_sess_2 = process_site(
        df_Juelich_sess_2_ml, "Jülich Session 2",
        rules=cluster_rules,
        cols=feature_cols
    )

    labels_Juelich_SD = process_site(
        df_Juelich_SD_ml, "Jülich Session SD",
        rules=cluster_rules,
        cols=feature_cols
    )

# %%
"""
SHAP beeswarm plot, SHAP-IQ bar plot, SHAP-IQ beeswarm, SHAP-IQ network plot for each site's each cluster
"""
def make_cluster_plots(
        *,
        labels:        np.ndarray | pd.Series,   # 1-D cluster ids
        explanation:   "shap.Explanation",       # per-sample SHAP result
        ivs:           list,                     # list[InteractionValues]
        feature_names: list[str],
        site_name:     str,
        save_dir:      str | None = None,
        min_max_order: tuple[int,int] = (1, 2),
        show:          bool = True,
        size_factor:   float = 5.0,
        max_display:   int = 20,
):
    """
    Plot SHAP & SHAP-iQ summaries per cluster.

    Notes
    -----
    • *labels* length must equal len(explanation) == len(ivs)
    • Figures are shown unless show=False; if save_dir is given, PNGs are saved.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    label_set = np.unique(labels)

    for cl in label_set:
        idx = np.where(labels == cl)[0]
        if idx.size == 0:
            continue

        # ── 1. average interaction matrix ─────────────────────────────────
        iv_avg = aggregate_interaction_values(
            [ivs[i] for i in idx],
            aggregation="mean"
        )

        # ── 2. network graph ──────────────────────────────────────────────
        fig, ax = shapiq.si_graph_plot(
            interaction_values = iv_avg,
            feature_names      = feature_names,
            min_max_order      = min_max_order,
            size_factor        = size_factor,
            show               = False,
        )
        plt.title(f"{site_name} – Cluster {cl} (n={len(idx)})", pad=10)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{site_name}_cl{cl}_network.png"),
                        dpi=300, bbox_inches="tight")
        if show: plt.show()
        plt.close()

        # ── 3. SHAP beeswarm ──────────────────────────────────────────────
        shap.plots.beeswarm(
            explanation[idx],
            clustering=False,
            show=False,
            max_display=max_display
        )
        plt.title(f"{site_name} – Cluster {cl} (n={len(idx)})", pad=10)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{site_name}_cl{cl}_shap_swarm.png"),
                        dpi=300, bbox_inches="tight")
        if show: plt.show()
        plt.close()

        # ── 4. SHAP-iQ bar plot ───────────────────────────────────────────
        shapiq.plot.bar_plot(
            [ivs[i] for i in idx],
            feature_names=feature_names,
            max_display=max_display,
            abbreviate=False,
            show=False
        )
        plt.title(f"{site_name} – Cluster {cl} (n={len(idx)})", pad=10)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{site_name}_cl{cl}_shapiq_bar.png"),
                        dpi=300, bbox_inches="tight")
        if show: plt.show()
        plt.close()

        print(f"[{site_name}] Cluster {cl} plots done.")

# %%
make_cluster_plots(
    labels        = labels_Liege.values,      # your rule-based cluster ids
    explanation   = explanation_train_set_test_Liege,
    ivs           = ivs_SHIP_Liege,           # list of InteractionValues
    feature_names = feature_cols,
    site_name     = "Liège",
    save_dir      = None,  # or None to skip saving
    show          = True
)

make_cluster_plots(
    labels = labels_VETSA.values,      # your rule-based cluster ids
    explanation = explanation_train_set_test_VETSA,
    ivs = ivs_SHIP_VETSA,           # list of InteractionValues
    feature_names = feature_cols,
    site_name = "San Diego",
    save_dir = None,  # or None to skip saving
    show = True
)

make_cluster_plots(
    labels = labels_Pitts.values,      # your rule-based cluster ids
    explanation = explanation_train_set_test_Pitts,
    ivs = ivs_SHIP_Pitts,           # list of InteractionValues
    feature_names = feature_cols,
    site_name = "Pittsburgh",
    save_dir = None,  # or None to skip saving
    show = True
)

if target == 'Memory':
    make_cluster_plots(
        labels = labels_KI.values,      # your rule-based cluster ids
        explanation = explanation_train_set_test_KI,
        ivs = ivs_SHIP_KI,           # list of InteractionValues
        feature_names = feature_cols,
        site_name = "Stockholm",
        save_dir = None,  # or None to skip saving
        show = True
    )

    make_cluster_plots(
        labels = labels_Juelich_sess_1.values,      # your rule-based cluster ids
        explanation = explanation_train_set_test_Juelich_sess_1,
        ivs = ivs_SHIP_Juelich_sess_1,           # list of InteractionValues
        feature_names = feature_cols,
        site_name = "Jülich Session 1",
        save_dir = None,  # or None to skip saving
        show = True
    )

    make_cluster_plots(
        labels = labels_Juelich_sess_2.values,      # your rule-based cluster ids
        explanation = explanation_train_set_test_Juelich_sess_2,
        ivs = ivs_SHIP_Juelich_sess_2,           # list of InteractionValues
        feature_names = feature_cols,
        site_name = "Jülich Session 2",
        save_dir = None,  # or None to skip saving
        show = True
    )

    make_cluster_plots(
        labels = labels_Juelich_SD.values,      # your rule-based cluster ids
        explanation = explanation_train_set_test_Juelich_SD,
        ivs = ivs_SHIP_Juelich_SD,           # list of InteractionValues
        feature_names = feature_cols,
        site_name = "Jülich Session SD",
        save_dir = None,  # or None to skip saving
        show = True
    )

# %%
