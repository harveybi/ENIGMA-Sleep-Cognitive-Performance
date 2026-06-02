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
# from cluster_explorer import Explainer as CE_explainer

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
# load SHIP_Trend dataset
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/{target}/SHIP_Trend/'

df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test', 'Memory_Test'])

"""
Define feature lists
"""
Sleep = ['PSG_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Dur', 'Self_Sleep_Eff', 'Depression_score']
Cov = ['Age_at_Scan', 'SEX', 'BMI']
APOE4 = ['APOE4']
TIV = 'EstimatedTotalIntraCranialVol'
sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']

columns = df_SHIP.columns.tolist()
Thickness_DK = columns[columns.index('lh_bankssts_thickness'):columns.index('rh_insula_thickness') + 1]
Thickness_Schaefer = columns[columns.index('LH_Vis_1_thickness'):columns.index('RH_Default_pCunPCC_9_thickness') + 1]
Area_DK = columns[columns.index('lh_bankssts_area'):columns.index('rh_insula_area') + 1]
Area_Schaefer = columns[columns.index('LH_Vis_1_area'):columns.index('RH_Default_pCunPCC_9_area') + 1]
Subcortical = columns[columns.index('Left-Lateral-Ventricle'):columns.index('CC_Anterior') + 1]

targets = ['Stroop_Test', 'Memory_Test']

"""
Data preprocessing
"""
df_SHIP_ml = utils.convert_units(df_SHIP, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].div(df_SHIP_ml[Thickness_DK].sum(axis=1), axis=0)
df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].div(df_SHIP_ml[Thickness_Schaefer].sum(axis=1), axis=0)
df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].div(df_SHIP_ml[Area_DK].sum(axis=1), axis=0)
df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].div(df_SHIP_ml[Area_Schaefer].sum(axis=1), axis=0)
df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].div(df_SHIP_ml['EstimatedTotalIntraCranialVol'], axis=0)

# NAI_Wordlist_Test transfer to accuracy
df_SHIP_ml['Memory_Test'] = df_SHIP_ml['Memory_Test'].apply(lambda x: x / 16) * 100

# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_SHIP_ml[APOE4] = df_SHIP_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

columns_to_object = ['SEX', 'APOE4']
df_SHIP_ml[columns_to_object] = df_SHIP_ml[columns_to_object].astype('category')

columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical
df_SHIP_ml[columns_to_float] = df_SHIP_ml[columns_to_float].astype('float64')

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
case_results_path = results_path + f"{feature_comb}/"
SHAP_path = case_results_path + 'SHAP/'
shapiq_path = case_results_path + 'shapiq/'

SHAP_clustering_path = case_results_path + 'SHAP_clustering/'
if not os.path.exists(SHAP_clustering_path):
    os.makedirs(SHAP_clustering_path)
silhouette_plot_path = SHAP_clustering_path + 'silhouette/'
if not os.path.exists(silhouette_plot_path):
    os.makedirs(silhouette_plot_path)

joblib_htcondor_path = "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/joblib_htcondor/SHAP_clustering/" + f"{target}/" + f"{feature_comb}"
if not os.path.exists(joblib_htcondor_path):
    os.makedirs(joblib_htcondor_path)
joblib_log_path =  "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/logs/SHAP_clustering/" + f"{target}/" + f"{feature_comb}"

# load saved shap value by pickle
print(f"\nLoading SHAP-IQ values for {feature_comb}, {target}.\n")
ivs_saved_path = shapiq_path + 'ivs_SHIP_SHIP.pkl'
with open(ivs_saved_path, 'rb') as f:
    ivs = pickle.load(f)

# load saved shap value by pickle
print(f"\nLoading SHAP values for {feature_comb}, {target}.\n")
shap_values_saved_path = SHAP_path + 'explanation_train_set_train.pkl'
with open(shap_values_saved_path, 'rb') as f:
    explanation_train_set_train = pickle.load(f)

explanation = explanation_train_set_train

# %%
shap.plots.beeswarm(explanation, clustering=False, show=False)
plt.tight_layout()
plt.show()
plt.close()

# %%
shap.plots.bar(explanation, max_display=20, show=False, clustering_cutoff=0.8)
plt.tight_layout()
plt.show()
plt.close()

# %%
shapiq.plot.bar_plot(ivs, feature_names=X, show=False, abbreviate=False, max_display=20)
plt.tight_layout()
plt.show()
plt.close()

shapiq.plot.beeswarm_plot(ivs, df_SHIP_ml[X], feature_names=X, abbreviate=False, show=False)
plt.tight_layout()
plt.show()
plt.close()

# %%
"""
Start of the SHAP value clustering
"""
print("\nExtracting SHAP values array and creating DataFrame...")

# Check if 'explanation' is the expected SHAP Explanation object
if hasattr(explanation, 'values') and hasattr(explanation, 'feature_names'):
    # Extract the SHAP values array
    shap_values_array = explanation.values
    # Extract the feature names from the explanation object (safer)
    feature_names_from_exp = explanation.feature_names

    # Optional: Sanity check shapes and feature name consistency
    print(f"Extracted SHAP values shape: {shap_values_array.shape}")
    print(f"Number of feature names in explanation object: {len(feature_names_from_exp)}")
    if shap_values_array.shape[1] != len(feature_names_from_exp):
        print(f"Warning: Mismatch between SHAP values columns ({shap_values_array.shape[1]}) and feature names count ({len(feature_names_from_exp)}). Using names from explanation object.")
        # Potentially add more robust error handling here if needed

    # Create the DataFrame using the extracted values and feature names
    try:
        shap_df = pd.DataFrame(shap_values_array, columns=feature_names_from_exp)
        print("\nSuccessfully created shap_df DataFrame:")
        print(shap_df.head())
    except Exception as e:
        print(f"\nError creating DataFrame even after extracting values: {e}")
        # Add fallback or further debugging if needed
        # Maybe try casting names to list: columns=list(feature_names_from_exp)

# Handle edge case where 'explanation' might be a list (e.g., multi-output)
elif isinstance(explanation, list) and len(explanation) > 0:
    print("Warning: Explanation object is a list. Assuming multi-output and using index 0.")
    # You might need to adjust the index [0] depending on your specific multi-output case
    first_output_exp = explanation[0]
    if hasattr(first_output_exp, 'values') and hasattr(first_output_exp, 'feature_names'):
        shap_values_array = first_output_exp.values
        feature_names_from_exp = first_output_exp.feature_names
        print(f"Extracted SHAP values shape (output 0): {shap_values_array.shape}")
        print(f"Number of feature names in explanation object (output 0): {len(feature_names_from_exp)}")
        try:
            shap_df = pd.DataFrame(shap_values_array, columns=feature_names_from_exp)
            print("\nSuccessfully created shap_df DataFrame (from output 0):")
            print(shap_df.head())
        except Exception as e:
            print(f"\nError creating DataFrame from list item: {e}")
    else:
        print("Error: Item in explanation list doesn't appear to be a valid SHAP Explanation object.")

# Handle case where 'explanation' might already be just a numpy array (less likely with modern SHAP)
elif isinstance(explanation, np.ndarray):
     print("Warning: 'explanation' appears to be a raw NumPy array. Using list X for columns.")
     shap_values_array = explanation
     if shap_values_array.shape[1] == len(X):
         feature_names_for_df = X
         try:
             shap_df = pd.DataFrame(shap_values_array, columns=feature_names_for_df)
             print("\nSuccessfully created shap_df DataFrame from NumPy array:")
             print(shap_df.head())
         except Exception as e:
             print(f"\nError creating DataFrame from numpy array: {e}")
     else:
         print(f"Error: NumPy array columns ({shap_values_array.shape[1]}) don't match length of X ({len(X)}).")

else:
    print("Error: Loaded 'explanation' object is not a recognized SHAP Explanation object or NumPy array.")

# %%
# if 'Brain' in feature_comb, reduce the dimensions of the brain features' SHAP values
if 'Brain' in feature_comb:
    # Select the brain features from the DataFrame
    brain_features = shap_df[Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical]

    # Apply UMAP to reduce dimensions to 10 components
    reducer = umap.UMAP(n_components=10, random_state=42)
    reduced_brain_features = reducer.fit_transform(brain_features)
    # Create a DataFrame with the reduced features
    reduced_brain_df = pd.DataFrame(reduced_brain_features, columns=[f'UMAP_{i+1}' for i in range(10)])
    # Concatenate the reduced features with the original DataFrame
    shap_df = pd.concat([shap_df, reduced_brain_df], axis=1)
    # Drop the original brain features
    shap_df = shap_df.drop(columns=Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical)
    # Print the new shape of the DataFrame
    print(f"New shape of shap_df after UMAP reduction: {shap_df.shape}")

# %%
"""
Clustering the SHAP matrix by Spectral Clustering.
"""
X_embed = shap_df.values

best_k = None
if 'Stroop' in target:
    best_k = 4
elif 'Memory' in target:
    best_k = 6

"""
Based on k = 2, 3, 4 compare the results for KMeans, Spectral Clustering, and GMM
"""
# z-score the SHAP values
# X_embed = StandardScaler().fit_transform(X_embed)

from itertools import cycle, islice

BASE_COLORS = [
    "#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
    "#984ea3", "#999999", "#e41a1c", "#dede00"
]


# def cluster_palette(labels):
#     """
#     Return a dict {label: hex‑color} using the fixed colour cycle.
#     Works for numeric or string labels.
#     """
#     uniq = np.unique(labels)              # preserves sort order
#     cols = list(islice(cycle(BASE_COLORS), len(uniq)))
#     return dict(zip(uniq.astype(str), cols))   # seaborn hue → string
#

def cluster_palette(labels):
    """
    Return a dict {label: hex‑color} using the fixed colour cycle.
    Works for numeric or string labels.
    """
    uniq = np.unique(labels)  # preserves sort order
    cols = list(islice(cycle(BASE_COLORS), len(uniq)))
    return dict(zip(uniq, cols))  # keep original label types

# %%
sns.set_theme(style="ticks", context="paper")

k_final = best_k
spec = SpectralClustering(
    n_clusters=k_final,
    affinity="nearest_neighbors",
    n_neighbors=15,
    assign_labels="kmeans",
    random_state=42
)
labels_final = spec.fit(X_embed).labels_.astype(str)

# gmm = GaussianMixture(
#     n_components=k_final,
#     covariance_type="full",  # "diag" or "tied" are alternatives
#     n_init=10,
#     max_iter=500,
#     random_state=42
# )
# labels_final = gmm.fit_predict(X_embed)

# km = KMeans(
#     n_clusters=best_k,
#     init="k-means++",
#     n_init=50,
#     max_iter=500,
#     random_state=42
# )
# labels_final = km.fit_predict(X_embed).astype(str)

labels_sorted = sorted(np.unique(labels_final))

# 1. UMAP embedding with the agreed parameters
reducer = umap.UMAP(
    n_components=2, n_neighbors=40, min_dist=0.1,
    metric="euclidean", random_state=42
)
emb = reducer.fit_transform(X_embed)

# %%
df_umap = pd.DataFrame({"U1": emb[:,0], "U2": emb[:,1],
                        "Cluster": labels_final})

plt.figure(figsize=(7,5))
sns.scatterplot(data=df_umap, x="U1", y="U2",
                hue="Cluster",
                palette=cluster_palette(labels_final),
                hue_order=labels_sorted,
                s=50, alpha=0.9, linewidth=0)
plt.title(f"UMAP – Spectral clustering (k={k_final})")
# plt.axis("off")
plt.legend(title="Cluster", frameon=False,
           bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
umap_path = f"{SHAP_clustering_path}spectral_final_k{k_final}_umap.png"
# plt.savefig(umap_path, dpi=300, bbox_inches="tight")
plt.show(); plt.close()
print(f"saved {umap_path}")

# 2. Feature‑vs‑SHAP scatter plots
scatter_dir = SHAP_clustering_path + "spectral_final_scatter/"
os.makedirs(scatter_dir, exist_ok=True)

for feat in ["Age_at_Scan", "PSG_Sleep_Dur", "Self_Sleep_Dur",
             "PSG_Sleep_Eff", "Self_Sleep_Eff"]:
    if feat not in df_SHIP_ml.columns or feat not in shap_df.columns:
        continue
    dfp = pd.DataFrame({
        "Original": df_SHIP_ml[feat].values,
        "SHAP":    shap_df[feat].values,
        "Cluster": labels_final
    })
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=dfp, x="Original", y="SHAP",
                    hue="Cluster",
                    palette=cluster_palette(labels_final),
                    hue_order=labels_sorted,
                    s=50, linewidth=0)
    plt.xlabel(f"{feat} (Original)")
    plt.ylabel("SHAP value")
    plt.title(f"{feat}: Original vs SHAP – Spectral (k={k_final})")
    plt.legend(title="Cluster", frameon=False,
               bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    out = f"{scatter_dir}{feat}_spectral_k{k_final}.png"
    # plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()
    print(f"saved {out}")

# %%
from matplotlib.ticker import FormatStrFormatter
fmt = FormatStrFormatter('%.0f')        # 10.0 → 10

feat_order = [
    "Age_at_Scan",
    "PSG_Sleep_Dur", "Self_Sleep_Dur",
    "PSG_Sleep_Eff", "Self_Sleep_Eff",
]
assert len(feat_order) == 5, "need exactly five feature names"

fig_dir  = SHAP_clustering_path
fig_name = f"spectral_k{k_final}_umap_plus_feats.png"

# -----------------------------------------------------------------
# build figure
# -----------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(8, 10),
    sharex=False, sharey=False, constrained_layout=True
)

# ---------------- panel (0,0)  UMAP ------------------------------
ax0 = axes[0, 0]
sns.scatterplot(
    x=emb[:, 0], y=emb[:, 1],
    hue=labels_final,
    palette=cluster_palette(labels_final),
    hue_order=labels_sorted,
    s=35, linewidth=0, alpha=0.9, ax=ax0,
    # legend=False,
)
ax0.set_title("UMAP – Spectral clustering", fontsize=10)
ax0.set_xlabel("UMAP 1"); ax0.set_ylabel("UMAP 2")

ax0.yaxis.set_major_formatter(fmt)

# ---------------- panels for the five features -------------------
for pos, feat in enumerate(feat_order, start=1):
    r, c = divmod(pos, 2)      # (row, col) in 0‑based grid
    ax  = axes[r, c]

    if feat not in shap_df.columns or feat not in df_SHIP_ml.columns:
        ax.set_visible(False)
        continue

    dfp = pd.DataFrame({
        "Orig":  df_SHIP_ml[feat].values,
        "SHAP":  shap_df[feat].values,
        "Cluster": labels_final
    })
    sns.scatterplot(
        data=dfp, x="Orig", y="SHAP",
        hue="Cluster",
        palette=cluster_palette(labels_final),
        hue_order=labels_sorted,
        s=35, linewidth=0, legend=False, ax=ax
    )
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel(f"{feat} (orig.)")
    ax.set_ylabel("SHAP")
    ax.yaxis.set_major_formatter(fmt)

# ---------------- one shared legend outside ----------------------
handles, labels = ax0.get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Cluster",
    loc="outside center right", frameon=False,
   # bbox_to_anchor=(1.02, 0.5)
)
# plt.tight_layout()
# ---------------- save / show -----------------------------------
# plt.savefig(os.path.join(fig_dir, fig_name), dpi=300,
#             bbox_inches="tight")
plt.show(); plt.close()
print("saved", fig_name)

# %%
features_to_colour = [
    "PSG_Sleep_Dur", "Self_Sleep_Dur",
    "PSG_Sleep_Eff", "Self_Sleep_Eff",
    "Age_at_Scan", 'BMI'
]

CMAP          = "RdBu_r"     # blue → white → red, reversed so red=high
VMIN, VMAX    = -4, 4        # same scale for every plot
feat_umap_dir = SHAP_clustering_path + "umap_by_feature/"
os.makedirs(feat_umap_dir, exist_ok=True)

# build the common coordinate DataFrame once
base_df = pd.DataFrame({"U1": emb[:, 0], "U2": emb[:, 1]})

for feat in features_to_colour:
    if feat not in shap_df.columns:
        print(f"{feat} missing – skipped."); continue

    vals = np.clip(shap_df[feat].values, VMIN, VMAX)   # optional clipping
    df   = base_df.copy()
    df["SHAP"] = vals

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(df["U1"], df["U2"],
                     c=vals, cmap="coolwarm",  # CMAP
                     vmin=VMIN, vmax=VMAX,
                     s=50, linewidths=0)

    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_ticks([VMIN, 0, VMAX])
    cbar.set_label(f"{feat} SHAP", rotation=270, labelpad=15)

    plt.title(f"UMAP coloured by {feat} SHAP")
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.tight_layout()

    out = f"{feat_umap_dir}{feat}_SHAP_on_UMAP.png"
    # plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()
    print(f"saved {out}")

# %%
rows = [
    ["Age_at_Scan", "BMI"],
    ["PSG_Sleep_Dur", "PSG_Sleep_Eff"],
    ["Self_Sleep_Dur", "Self_Sleep_Eff"],
]
CMAP           = "RdBu_r"
VMIN, VMAX     = -4, 4
feat_umap_path = SHAP_clustering_path + "umap_SHAP_grid.png"

# ---------------------------------------------------------------
# common coordinates
# ---------------------------------------------------------------
base_df = pd.DataFrame({"U1": emb[:, 0], "U2": emb[:, 1]})
# xlim = (base_df["U1"].min(), base_df["U1"].max())
# ylim = (base_df["U2"].min(), base_df["U2"].max())

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(7, 8),
    sharex=True, sharey=True,
    # constrained_layout=True
)

for r, row in enumerate(rows):
    for c, feat in enumerate(row):
        ax = axes[r, c]
        if feat not in shap_df.columns:
            ax.set_visible(False)
            continue

        vals = np.clip(shap_df[feat].values, VMIN, VMAX)
        sc = ax.scatter(
            base_df["U1"], base_df["U2"],
            c=vals, cmap=CMAP,
            vmin=VMIN, vmax=VMAX,
            s=35, linewidths=0
        )
        ax.set_title(feat, fontsize=11)
        # ax.set_xlim(xlim); ax.set_ylim(ylim)
        # ax.set_xlabel("UMAP 1" if r == 2 else "")
        # ax.set_ylabel("UMAP 2" if c == 0 else "")
        ax.tick_params(length=3)

# ---------------------------------------------------------------
# one shared colour‑bar
# ---------------------------------------------------------------
# cbar = fig.colorbar(
#     sc, ax=axes, orientation="vertical",
#     fraction=0.02, pad=0.02
# )
# cbar.set_ticks([VMIN, 0, VMAX])
# cbar.set_label("SHAP value", rotation=270, labelpad=15)

fig.supxlabel("UMAP 1", y=0.02, fontsize=10)   # y in figure coords
fig.supylabel("UMAP 2", x=0.02, fontsize=10)   # x in figure coords
plt.tight_layout()
# plt.savefig(feat_umap_path, dpi=300, bbox_inches="tight")
plt.show(); plt.close()
print(f"saved {feat_umap_path}")

# %%
"""
Clustering rules by 1. SkopeRules 2. ClusterExplorer
"""
# ------------------------------------------------------------------
# (A)  Feature matrix  X
# ------------------------------------------------------------------
# → use the SHAP-value DataFrame you built earlier
# df_X = shap_df.copy()
# df_SHIP.reset_index(drop=True)  # reset index
df_X = df_SHIP_ml[X].reset_index(drop=True)

# ------------------------------------------------------------------
# (B)  Cluster labels  y_clust
# ------------------------------------------------------------------
# → use the Spectral-Clustering result you stored in labels_final
y_clust = pd.Series(labels_final, name="cluster").astype(str)

print(df_X.shape, y_clust.value_counts().to_dict())

# %%
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

sr_models = fit_skope_by_cluster(df_X, y_clust)

# %%
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
# cluster_explainer
# X        = shap_df.copy()           # ← your SHAP–value DataFrame
# clusters = y_clust            # ← Spectral-clustering labels (array-like)
#
# cluster_explainer = CE_explainer(
#     df_X,                               # feature matrix (pandas or ndarray)
#     clusters,                        # 1-d array / Series of cluster ids
# )
#
# df_cluster_explore = cluster_explainer.generate_explanations(
#     # coverage_threshold     = 0.80,   # ≥ 80 % of the cluster’s points
#     # conciseness_threshold  = 0.20,   # ≤ 20 % of *all* features used on avg.
#     # separation_threshold   = 0.20,   # ≤ 20 % contamination from other clusters
#     # p_value                = int(1/0.20)  # = 5  (Fisher test significance)
# )

# %%
"""
Subject selection based on clusters
"""
from sklearn.metrics import pairwise_distances

def cluster_representatives(X, labels, metric="euclidean"):
    """
    Return a dict {cluster_id: index_of_most_representative_member}.
    """
    reps = {}
    for cl in np.unique(labels):
        if cl == "-1":          # skip noise if you have it
            continue
        idx   = np.where(labels == cl)[0]           # rows in this cluster
        X_cl  = X[idx]

        # OPTION A ─ centroid distance (fast, good for >500 pts)
        # centre = X_cl.mean(axis=0, keepdims=True)
        # dists  = pairwise_distances(X_cl, centre, metric=metric).ravel()
        # reps[cl] = idx[dists.argmin()]

        # OPTION B ─ medoid (small clusters, exact)
        D   = pairwise_distances(X_cl, metric=metric)
        med = D.mean(axis=1).argmin()
        reps[cl] = idx[med]

    return reps

rep_idx = cluster_representatives(X_embed, labels_final, metric="euclidean")
print("Representative subject indices per cluster:", rep_idx)

# %%
def reps_by_mean_shap(shap_df, labels, use_abs=True):
    """
    Parameters
    ----------
    shap_df : pandas.DataFrame
        Same shape/order as the matrix you clustered (rows = subjects).
    labels  : 1‑D array‑like
        Cluster labels for each subject.
    use_abs : bool
        True  → use mean(|SHAP|)  (total magnitude, ignores sign)
        False → use mean(SHAP)    (signed average).
    Returns
    -------
    reps : dict  {cluster_id : row‑index of the chosen subject}
    """
    reps = {}
    for cl in np.unique(labels):
        if cl == "-1":          # skip noise if present
            continue
        idx = np.where(labels == cl)[0]      # indices in this cluster
        sub_df = shap_df.iloc[idx]

        if use_abs:
            scores = sub_df.abs().mean(axis=1)   # mean |SHAP| per subject
        else:
            scores = sub_df.mean(axis=1)         # signed mean

        reps[cl] = scores.idxmax()               # DataFrame index → row #
    return reps


# ------------------------------------------------------------------
# example usage with your Spectral labels
# ------------------------------------------------------------------
rep_idx_mag = reps_by_mean_shap(shap_df, labels_final, use_abs=True)
print("Representative (largest mean |SHAP|) per cluster:", rep_idx_mag)

# %%
net_dir = SHAP_clustering_path + "rep_networks/"
os.makedirs(net_dir, exist_ok=True)

feature_names = X

for cl, idx in rep_idx_mag.items():  #  rep_idx or rep_idx_mag

    # -------------------------------
    # extract that subject's interaction matrix
    # ivs is [sample, i, j]  (assume symmetric, diagonal = main effect)
    # -----------------------------------------------------------------
    interaction_values = ivs[idx]

    # fig = shapiq.network_plot(
    #     first_order_values = interaction_values.get_n_order_values(1),
    #     second_order_values= interaction_values.get_n_order_values(2),
    #     feature_names      = feature_names,
    #     draw_legend = False
    # )

    fig = shapiq.network_plot(
        interaction_values=interaction_values,
        feature_names=feature_names,
    )

    out = f"{net_dir}cluster_{cl}_rep_network.png"
    # title
    plt.title(f"Cluster {cl} representative subject", fontsize=14)
    plt.tight_layout()
    # save figure
    # plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"saved {out}")

# %%
"""
Average first_order_values, second_order_values base on each clusters. Then make shap interaction nextwork plot.
"""
net_dir = SHAP_clustering_path + "cluster_avg_networks/"
os.makedirs(net_dir, exist_ok=True)

feature_names = X
label_set = [cl for cl in np.unique(labels_final) if cl != "-1"]

# %%
"""
SHAP-IQ updated, they provided a new function to calculate the average interaction values. 
shapiq.interaction_values.aggregate_interaction_values
But the network plot made by the average ivs by the aggregate_interaction_values is not the same as before.
Here will check whether the average ivs calculated by the aggregate_interaction_values is same as the average ivs calculated by the previous method.
"""
# use first cluster as an example
cl = label_set[0]
idx = np.where(labels_final == cl)[0]       # subjects in this cluster
if len(idx) == 0:
    raise ValueError(f"No subjects found in cluster {cl}")

# -------------------------------------------------------------
# 1st‑ and 2nd‑order arrays for every subject, then mean
# -------------------------------------------------------------
first_list  = []
second_list = []
for s in idx:
    iv = ivs[s]
    first_list.append(iv.get_n_order_values(1))   # shape (F,)
    second_list.append(iv.get_n_order_values(2))    # shape (F,F)
first_avg  = np.mean(first_list,  axis=0)       # (F,)
second_avg = np.mean(second_list, axis=0)       # (F,F)
# -------------------------------------------------------------
# use the new function to calculate the average ivs
ivs_list = []
for s in idx:
    iv = ivs[s]
    ivs_list.append(iv)  # wrap the 2‑D matrix
iv_avg = shapiq.interaction_values.aggregate_interaction_values(ivs_list, aggregation="mean")  # mean over subjects

# %%
# check if the average ivs are the same
if np.allclose(first_avg, iv_avg.get_n_order_values(1)) and np.allclose(second_avg, iv_avg.get_n_order_values(2)):
    print("The average interaction values are the same.")
else:
    print("The average interaction values are different.")
    print("First order values are different:", not np.allclose(first_avg, iv_avg.get_n_order_values(1)))
    print("Second order values are different:", not np.allclose(second_avg, iv_avg.get_n_order_values(2)))

# %%
# network plot on the averaged orders
abs_vals = np.abs(iv_avg.values)
lo, hi   = np.percentile(abs_vals, [5, 95])   # 5-th to 95-th percentile

fig, ax = shapiq.si_graph_plot(  # new signature :contentReference[oaicite:1]{index=1}
    interaction_values=iv_avg,
    feature_names=feature_names,
    show=False,
    min_max_order=(1, 2),
    size_factor=5.0,
    node_size_scaling=0.8,
    min_max_interactions = (-hi, hi),
)

plt.title(f"Cluster {cl} – mean interaction network (n={len(idx)})",
            fontsize=13, pad=10)
plt.tight_layout()
plt.show(); plt.close()

# %%
for cl in label_set:
    idx = np.where(labels_final == cl)[0]       # subjects in this cluster
    if len(idx) == 0:
        continue

    # -------------------------------------------------------------
    # 1st‑ and 2nd‑order arrays for every subject, then mean
    # -------------------------------------------------------------
    # first_list  = []
    # second_list = []
    # for s in idx:
    #     iv = ivs[s]
    #     first_list.append(iv.get_n_order_values(1))   # shape (F,)
    #     second_list.append(iv.get_n_order_values(2))    # shape (F,F)
    #
    # first_avg  = np.mean(first_list,  axis=0)
    # second_avg = np.mean(second_list, axis=0)
    ivs_list = []
    for s in idx:
        iv = ivs[s]
        ivs_list.append(iv)  # wrap the 2‑D matrix

    iv_avg = aggregate_interaction_values(ivs_list, aggregation="mean")  # mean over subjects :contentReference[oaicite:0]{index=0}
    # TODO: Double check if it's correct
    # -------------------------------------------------------------
    # network plot on the averaged orders
    # -------------------------------------------------------------
    # fig = shapiq.network_plot(
    #     first_order_values  = first_avg,
    #     second_order_values = second_avg,
    #     feature_names       = feature_names,
    #     draw_legend         = False,
    # )
    # abs_vals = np.abs(iv_avg.values)
    # lo, hi = np.percentile(abs_vals, [5, 95])  # 5-th to 95-th percentile

    fig, ax = shapiq.si_graph_plot(  # new signature :contentReference[oaicite:1]{index=1}
        interaction_values=iv_avg,
        feature_names=feature_names,
        show=False,
        min_max_order=(1, 2),
        size_factor=5.0,
        # node_size_scaling=0.8,
        # min_max_interactions = (-hi, hi),
    )

    plt.title(f"Cluster {cl} – mean interaction network (n={len(idx)})",
              fontsize=13, pad=10)
    plt.tight_layout()

    out = f"{net_dir}cluster_{cl}_avg_network.png"
    # save figure
    # plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    expl_cluster = explanation[idx]  # simple slicing
    shap.plots.beeswarm(expl_cluster,  # SHAP >=0.43 API :contentReference[oaicite:1]{index=1}
                        clustering=False, show=False)
    plt.tight_layout()
    plt.show()
    plt.close()

    # iv_list = [ivs[s] for s in idx]
    shapiq.plot.bar_plot(  # wrapper around SHAP bar :contentReference[oaicite:2]{index=2}
        ivs_list,
        feature_names=feature_names,
        abbreviate=False,
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.show()
    plt.close()

    print(f"saved {out}")

# %%
# make beeswarm plot for each cluster only
def beeswarm_cluster(explanation, labels, cluster_id, feature_names=None):
    """
    Create a beeswarm plot for a specific cluster.
    Parameters:
    - explanation: SHAP Explanation object or list of explanations.
    - labels: Cluster labels for each sample.
    - cluster_id: The cluster ID to plot.
    - feature_names: Optional list of feature names for the plot.
    """
    if isinstance(explanation, list):
        expl_cluster = explanation[labels == cluster_id]
    else:
        expl_cluster = explanation[labels == cluster_id]

    shap.plots.beeswarm(expl_cluster,  # SHAP >=0.43 API
                        clustering=False, show=False,
                        feature_names=feature_names)
    plt.title(f"Beeswarm plot for Cluster {cluster_id}")
    plt.tight_layout()
    plt.show()

# Example usage for a specific cluster
for cl in np.unique(labels_final):
    if cl == "-1":  # skip noise cluster if present
        continue
    print(f"Beeswarm plot for Cluster {cl}")
    beeswarm_cluster(explanation, labels_final, cl, feature_names=X)

# %%
# describe each cluster's original feature values, robust for float and categorical features
def describe_cluster_features(df, labels):
    """
    Describe the original feature values for each cluster.
    Returns a DataFrame with cluster labels as index and descriptive statistics.
    """
    desc = df.copy()
    desc["Cluster"] = labels
    desc = desc.groupby("Cluster").describe().T  # transpose for better readability
    return desc

cluster_desc = describe_cluster_features(df_SHIP_ml[X], labels_final)
print("\nCluster feature descriptions:")
print(cluster_desc)

# %%
def describe_cluster_features(df, labels, categorical_cols=None):
    df = df.copy()
    df["Cluster"] = labels
    cluster_groups = df.groupby("Cluster")

    # Numeric description
    numeric_desc = cluster_groups.describe().T

    # Categorical summary: value counts per cluster normalized (i.e., proportions)
    cat_summary = {}
    if categorical_cols is not None:
        for col in categorical_cols:
            counts = cluster_groups[col].value_counts(normalize=False).unstack().fillna(0)
            counts.columns = [f"{col}={val}" for val in counts.columns]
            cat_summary[col] = counts

        cat_desc = pd.concat(cat_summary.values(), axis=1)
    else:
        cat_desc = pd.DataFrame()

    return numeric_desc, cat_desc

# Example usage
categorical_cols = ['SEX']
num_desc, cat_desc = describe_cluster_features(df_SHIP_ml[X], labels_final, categorical_cols=categorical_cols)

# %%
from pathlib import Path
from typing import Sequence, Union, Optional

def build_cluster_feature_table(
    df: pd.DataFrame,
    labels: Union[pd.Series, np.ndarray, Sequence],
    *,
    numeric_cols: Optional[list[str]] = None,
    categorical_cols: Optional[list[str]] = None,
    decimals: int = 2,
    save_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Build a summary table (Mean±SD, Range for numerics; counts for categoricals)
    for each cluster. Optionally write to CSV.

    Parameters
    ----------
    df              : DataFrame of original features (no cluster column yet)
    labels          : 1-D iterable of cluster labels (same length as df)
    numeric_cols    : columns to treat as numeric.  If None → infer all numeric
    categorical_cols: columns to treat as categorical.  If None → []
    decimals        : # decimals for floats
    save_path       : Full path *including filename* for CSV.
                      If None, the file is **not** saved; the table is returned.

    Returns
    -------
    out_df          : Multi-index DataFrame in “feature × statistic” format
    """
    df_ = df.copy()
    df_["Cluster"] = labels

    if categorical_cols is None:
        categorical_cols = []
    if numeric_cols is None:
        numeric_cols = df_.select_dtypes(include=[np.number]).columns.difference(categorical_cols)

    clusters = sorted(df_["Cluster"].unique())
    n_per_cluster = df_.groupby("Cluster").size()
    col_headers = [f"Cluster {c} (N={n_per_cluster[c]})" for c in clusters]

    rows, idx = [], []

    g = df_.groupby("Cluster")
    # ----- numeric features -------------------------------------------------
    for col in numeric_cols:
        means = g[col].mean()
        stds  = g[col].std()
        mins  = g[col].min()
        maxs  = g[col].max()

        rows.append([f"{means[c]:.{decimals}f} ± {stds[c]:.{decimals}f}" for c in clusters])
        idx.append((col, "Mean ± SD"))

        rows.append([f"{mins[c]:.{decimals}f}–{maxs[c]:.{decimals}f}" for c in clusters])
        idx.append((col, "Range"))

    # ----- categorical features --------------------------------------------
    for col in categorical_cols:
        levels = (list(df_[col].cat.categories)
                  if pd.api.types.is_categorical_dtype(df_[col])
                  else sorted(df_[col].unique()))
        for lvl in levels:
            counts = g[col].apply(lambda s, v=lvl: (s == v).sum())
            rows.append([int(counts[c]) for c in clusters])
            idx.append((col, str(lvl)))

    out_df = pd.DataFrame(
        rows,
        columns=col_headers,
        index=pd.MultiIndex.from_tuples(idx, names=["Feature", "Statistic"])
    )

    # ----- write or print ---------------------------------------------------
    if save_path is not None:
        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(save_path)
        print(f"✓ Saved table to: {save_path}")
    else:
        # fall-back: print a preview in the console
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(out_df)

    return out_df


# summary_df = write_cluster_feature_csv(
#     df_SHIP_ml[X],          # your feature DataFrame
#     labels_final,           # 1-D array of cluster assignments
#     numeric_cols=[
#         "Age_at_Scan", "BMI",
#         "Self_Sleep_Dur",  # ← change to your actual column names
#         "PSG_Sleep_Dur",
#         "Self_Sleep_Eff",
#         "PSG_Sleep_Eff",
#         "Depression_score"
#     ],
#     categorical_cols=["SEX"],
#     filename="cluster_feature_summary.csv"
# )

summary_df = build_cluster_feature_table(
    df_SHIP_ml[X],
    labels_final,
    # numeric_cols=[
    #     "Age_at_Scan",
    #     "BMI",
    #     "Self_Sleep_Dur",  # ← change to your actual column names
    #     "PSG_Sleep_Dur",
    #     "Self_Sleep_Eff",
    #     "PSG_Sleep_Eff",
    #     "Depression_score"
    # ],
    numeric_cols=[
        "Age_at_Scan",
        "PSG_Sleep_Eff",  # ← change to your actual column names
        "Depression_score",
        "PSG_Sleep_Dur",
        "BMI",
        "Self_Sleep_Dur",
        "Self_Sleep_Eff"
    ],
    categorical_cols=["SEX"],
    save_path=case_results_path + "cluster_feature_summary.csv"
)


# %%
"""
network plot based on all subjects' average
"""
# ---------------------------------------------------------------------
first_all  = []
second_all = []

for s in range(len(ivs)):                      # 0 … n_subjects‑1
    iv = ivs[s]      # wrap the 2‑D matrix
    first_all.append(iv.get_n_order_values(1))     # shape (F,)
    second_all.append(iv.get_n_order_values(2))    # shape (F, F)

first_all_avg  = np.mean(first_all,  axis=0)       # (F,)
second_all_avg = np.mean(second_all, axis=0)       # (F, F)

iv_all_avg = shapiq.interaction_values.aggregate_interaction_values(ivs, aggregation="mean")

# check if the average ivs are the same
if np.allclose(first_all_avg, iv_all_avg.get_n_order_values(1)) and np.allclose(second_all_avg, iv_all_avg.get_n_order_values(2)):
    print("The average interaction values are the same.")
else:
    print("The average interaction values are different.")
    print("First order values are different:", not np.allclose(first_all_avg, iv_all_avg.get_n_order_values(1)))
    print("Second order values are different:", not np.allclose(second_all_avg, iv_all_avg.get_n_order_values(2)))

# %%
# fig = shapiq.network_plot(
#     first_order_values  = first_all_avg,
#     second_order_values = second_all_avg,
#     feature_names       = feature_names,             # same feature order
#     draw_legend         = False,
# )
fig = shapiq.si_graph_plot(  # new signature :contentReference[oaicite:1]{index=1}
    interaction_values=iv_all_avg,
    feature_names=feature_names,
    show=False,
    min_max_order=(1, 2),
    size_factor=5.0,
)

# TODO: Consider to use SI plot

plt.title("Mean 1st/2nd order interactions (all subjects)",
          fontsize=13, pad=10)
plt.tight_layout()

out_all = SHAP_clustering_path + "network_mean_all_subjects.png"
# save
plt.savefig(out_all, dpi=300, bbox_inches="tight")
plt.show(); plt.close()
print(f"saved {out_all}")
