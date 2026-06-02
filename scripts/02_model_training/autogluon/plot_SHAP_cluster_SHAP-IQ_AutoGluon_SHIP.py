import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import AutoGluon_pipeline
from AutoGluon_pipeline import AdMLPipeline

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import shap
import shapiq
import pickle
from datetime import datetime

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array

import umap.umap_ as umap

from joblib import dump, load

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)

import seaborn as sns
sns.set_context("paper")

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

columns = df_SHIP.columns.tolist()
Thickness_DK = columns[columns.index('lh_bankssts_thickness'):columns.index('rh_insula_thickness') + 1]
Thickness_Schaefer = columns[columns.index('LH_Vis_1_thickness'):columns.index('RH_Default_pCunPCC_9_thickness') + 1]
Area_DK = columns[columns.index('lh_bankssts_area'):columns.index('rh_insula_area') + 1]
Area_Schaefer = columns[columns.index('LH_Vis_1_area'):columns.index('RH_Default_pCunPCC_9_area') + 1]
Subcortical = columns[columns.index('Left-Lateral-Ventricle'):columns.index('CC_Anterior') + 1]

targets = ['Stroop_Test', 'Memory_Test']

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
shapiq.plot.bar_plot(ivs, feature_names=X, show=False, abbreviate=False, max_display=20)
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
1. Use GMM compute BIC to select the best k
2. Based on best k compare the results for KMeans, Spectral Clustering, and GMM
3. Compare the results of KMeans, Spectral Clustering, and GMM by Gap statistic and silhouette score plots
"""

"""
Use GMM compute BIC to select the best k
"""
# ----------------------------------------------------------------------
# 1. Define the data for clustering (SHAP values)
# Using all SHAP features for clustering
X_embed = shap_df.values

# ----------------------------------------------------------------------
# 2. Grid-search for k over the recommended range (2 to 15)
k_range = range(2, 16)  # Test k from 2 up to 15
silhouette_by_k  = {}
bic_by_k         = {}
nll_by_k         = {}  # negative log‑likelihood (≈ “elbow”)

print("Calculating GMM silhouette & BIC for k in", list(k_range)) # Show the list
for k in k_range:
    # ── fit GMM ─────────────────────────────────────────────────────────
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",  # "diag" or "tied" are alternatives
        n_init=10,
        max_iter=500,
        random_state=42
    )
    labels_k = gmm.fit_predict(X_embed)

    # log‑likelihood per sample → negative log‑likelihood total
    nll = -gmm.score(X_embed) * X_embed.shape[0]
    nll_by_k[k] = nll

    # BIC (lower = better)
    bic = gmm.bic(X_embed)
    bic_by_k[k] = bic

    print(f'k = {k} → NLL = {nll:0.1f}', end='')

    # ── silhouette & per‑k silhouette plot ─────────────────────────────
    if len(set(labels_k)) > 1:
        sil = silhouette_score(X_embed, labels_k)
        silhouette_by_k[k] = sil
        print(f' | avg silhouette = {sil:0.3f}')

        # individual silhouette diagram
        sil_vals = silhouette_samples(X_embed, labels_k)
        fig, ax1 = plt.subplots(figsize=(6, 4))
        y_lower = 10
        for i in range(k):
            ith_vals = sil_vals[labels_k == i]
            ith_vals.sort()
            size_i = ith_vals.size
            y_upper = y_lower + size_i

            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, ith_vals,
                facecolor=color, edgecolor=color, alpha=0.7
            )
            ax1.text(-0.05, y_lower + 0.5 * size_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title(f"Silhouette plot — GMM, k = {k}")
        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_yticks([])
        ax1.axvline(x=sil, color="red", ls="--", label=f"avg = {sil:0.3f}")
        ax1.set_xlim([-0.1, 1])
        ax1.legend(loc="lower right", frameon=False)
        plt.tight_layout()

        sil_path = silhouette_plot_path + f"silhouette_gmm_k{k}.png"
        plt.savefig(sil_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"  → silhouette figure saved to {sil_path}")
    else:
        silhouette_by_k[k] = -1
        print(" | only one cluster – silhouette undefined")

    print(f' | BIC = {bic:0.1f}')

# %%
"""
Based on best k compare the results for KMeans, Spectral Clustering, and GMM
"""
def _argmin(d):
    """Return key with smallest value; None if dict empty or all NaNs."""
    if not d:
        return None
    # Filter out nan / inf in case GMM failed for a k
    valid = {k: v for k, v in d.items() if np.isfinite(v)}
    return min(valid, key=valid.get) if valid else None

# select best k from BIC
best_k = _argmin(bic_by_k)

# Fallback to silhouette if BIC missing
if best_k is None:
    print("\nWarning: No valid BIC values – falling back to silhouette.")
    if silhouette_by_k and any(v != -1 for v in silhouette_by_k.values()):
        best_k = max(silhouette_by_k, key=silhouette_by_k.get)
        print(f"Chosen k = {best_k} (highest silhouette score: {silhouette_by_k[best_k]:.3f})")
    else:
        best_k = 3
        print("\nError: No valid silhouette scores either. Using default k = 3.")
else:
    print(f"\nChosen k = {best_k} (lowest BIC: {bic_by_k[best_k]:.1f})")

# %%
best_k = 6  # use this when the best_k selected automatically is not good enough

algo_dict = {
    "kmeans": KMeans(
        n_clusters=best_k,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=42
    ),
    "spectral": SpectralClustering(
        n_clusters=best_k,
        affinity="nearest_neighbors",  # nearest‑neighbor graph works well for medium‑large n
        n_neighbors=15,                # tweak if your data are very dense/sparse
        random_state=42,
        assign_labels="kmeans"         # gives crisp labels
    ),
    "gmm": GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        n_init=10,
        max_iter=500,
        random_state=42
    )
}

results = []
for name, model in algo_dict.items():
    print(f"\nFitting {name} …")
    if name == "gmm":
        labels_pred = model.fit_predict(X_embed)
        # Log‑likelihood and BIC have opposite direction to DBI,
        # so just store BIC for separate inspection
        bic_val = model.bic(X_embed)
    else:
        labels_pred = model.fit(X_embed).labels_
        bic_val = np.nan

    # Guard against algorithms that collapse into one cluster
    if len(np.unique(labels_pred)) < 2:
        print(f"{name} produced only one cluster — metrics undefined.")
        sil, chs, db = np.nan, np.nan, np.nan
    else:
        sil = silhouette_score(X_embed, labels_pred)
        chs = calinski_harabasz_score(X_embed, labels_pred)
        db  = davies_bouldin_score(X_embed, labels_pred)   # lower is better

    results.append(dict(
        Algorithm      = name,
        Silhouette     = sil,
        CalinskiHarabasz = chs,
        DaviesBouldin  = db,
        BIC            = bic_val
    ))

# Turn into a nice DataFrame for easy reading
metric_df = pd.DataFrame(results).set_index("Algorithm")
print("\n=== Clustering quality comparison (k = {}) ===".format(best_k))
print(metric_df.sort_values("Silhouette", ascending=False))

# If you want to save it for later inspection:
metric_csv = SHAP_clustering_path + f"clustering_metrics_k{best_k}.csv"
metric_df.to_csv(metric_csv)
print(f"\nMetric table saved to: {metric_csv}")

# Optionally: pick the “winner” (highest silhouette, tie‑break by CHS)
valid_df = metric_df.dropna(subset=["Silhouette"])
if not valid_df.empty:
    winner = valid_df.sort_values(
        ["Silhouette", "CalinskiHarabasz"], ascending=[False, False]
    ).index[0]
    print(f"\nBest algorithm by Silhouette → {winner}")
else:
    print("\nNo valid silhouettes to choose a winner.")

# %%
# ── keep the labels we already computed in the comparison step ─────────
labels_dict = {}          # {"kmeans": array([...]), "spectral": …, "gmm": …}
for name, model in algo_dict.items():
    if name == "gmm":
        labels_dict[name] = model.predict(X_embed)
    else:
        labels_dict[name] = model.labels_        # they were fit already
#
# # ── helper to make *all three* figures for one algorithm ───────────────
# def visualise_algo(algo_name, lbls):
#     print(f"\nCreating visualisations for «{algo_name}»")
#
#     param_grid = [
#         # ――― a balanced starter set ―――
#         {"metric": "euclidean", "n_neighbors": 15, "min_dist": 0.1},
#         {"metric": "euclidean", "n_neighbors": 30, "min_dist": 0.1},
#         {"metric": "manhattan", "n_neighbors": 30, "min_dist": 0.1},
#         {"metric": "chebyshev", "n_neighbors": 30, "min_dist": 0.1},
#         # add / remove rows as you like
#     ]
#
#     for p in param_grid:
#         metric = p["metric"]
#         nn = p["n_neighbors"]
#         md = p["min_dist"]
#
#         reducer = umap.UMAP(
#             n_components=2,
#             n_neighbors=nn,
#             min_dist=md,
#             metric=metric,
#             random_state=42
#         )
#         emb = reducer.fit_transform(X_embed)
#
#         df_umap = pd.DataFrame({
#             "UMAP‑1": emb[:, 0],
#             "UMAP‑2": emb[:, 1],
#             "Cluster": lbls.astype(str)
#         })
#
#         plt.figure(figsize=(7, 5))
#         sns.scatterplot(
#             data=df_umap,
#             x="UMAP‑1", y="UMAP‑2",
#             hue="Cluster",
#             palette="viridis",
#             s=50, alpha=0.9, linewidth=0
#         )
#
#         title = (f"UMAP ({algo_name}, k={best_k})  |  "
#                  f"metric={metric}, n_neighbors={nn}, min_dist={md}")
#         plt.title(title)
#         plt.axis("off")
#         plt.legend(title="Cluster", frameon=False,
#                    bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#
#         umap_save_path = SHAP_clustering_path + "umap/"
#         if not os.path.exists(umap_save_path):
#             os.makedirs(umap_save_path)
#         fname = umap_save_path + f"{algo_name}_k{best_k}_{metric}_nn{nn}_md{md}.png"
#         plt.savefig(fname, dpi=300, bbox_inches="tight")
#         plt.show()
#         plt.close()
#         print(f"saved {fname}")
#
# # ── run the helper for every algorithm ─────────────────────────────────
# for algo_name, lbls in labels_dict.items():
#     visualise_algo(algo_name, lbls)

# %%
def per_algo_diagnostics(algo_name, labels, k_used):
    """
    Draw detailed silhouette stats and two feature‑vs‑SHAP scatter plots
    for one clustering result.

    Parameters
    ----------
    algo_name : str   (“kmeans”, “spectral”, “gmm”, …)
    labels    : 1‑D ndarray of shape (n_samples,)
    k_used    : int   number of clusters you asked the algorithm to fit
    """
    # ── 4. Detailed silhouette ─────────────────────────────────────────
    if len(np.unique(labels)) > 1:
        sil_global = silhouette_score(X_embed, labels)
        sil_indiv  = silhouette_samples(X_embed, labels)

        print(f"\n[{algo_name}] Avg. silhouette (k={k_used}): {sil_global:0.3f}")
        for lab in np.unique(labels):
            vals = sil_indiv[labels == lab]
            print(f"  – Cluster {lab}: N={vals.size:3d} | μ={vals.mean():0.3f} | σ={vals.std():0.3f}")

        # Optional: detailed silhouette bar plot
        fig, ax = plt.subplots(figsize=(6, 4))
        y_lower = 10
        for i in range(k_used):
            ith_vals = sil_indiv[labels == i]
            ith_vals.sort()
            y_upper = y_lower + ith_vals.size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                             facecolor=cm.nipy_spectral(i / k_used), alpha=0.7)
            ax.text(-0.05, y_lower + 0.5*ith_vals.size, str(i))
            y_lower = y_upper + 10
        ax.axvline(sil_global, color="red", ls="--", label=f"avg={sil_global:0.3f}")
        ax.set_title(f"Silhouette – {algo_name} (k={k_used})")
        ax.set_xlabel("Silhouette coefficient"); ax.set_yticks([])
        ax.set_xlim([-0.1, 1]); ax.legend()
        plt.tight_layout()
        fname = f"{SHAP_clustering_path}{algo_name}_silhouette_k{k_used}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  → silhouette figure saved: {fname}")

    else:
        print(f"[{algo_name}] Only one cluster – silhouettes undefined.")

    # ── 5. Two feature scatter plots ──────────────────────────────────
    for feat in ["PSG_Sleep_Dur", "Age_at_Scan"]:
        if feat not in df_SHIP.columns or feat not in shap_df.columns:
            print(f"Feature '{feat}' not found – skipping.")
            continue

        df_plot = pd.DataFrame({
            "Original": df_SHIP[feat].values,
            "SHAP":    shap_df[feat].values,
            "Cluster": labels.astype(str)
        })

        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df_plot, x="Original", y="SHAP",
                        hue="Cluster", palette="viridis", s=50, linewidth=0)
        plt.xlabel(f"{feat} (Original)"); plt.ylabel(f"SHAP value")
        plt.title(f"{feat}: Original vs. SHAP  ({algo_name}, k={k_used})")
        plt.legend(title="Cluster", frameon=False,
                   bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fname = f"{SHAP_clustering_path}{algo_name}_{feat}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.show(); plt.close()
        print(f"  → scatter figure saved: {fname}")

# ----------------------------------------------------------------------
# Run diagnostics for every algorithm you compared earlier
# ----------------------------------------------------------------------
for algo_name, lbls in labels_dict.items():     # labels_dict created in comparison step
    per_algo_diagnostics(algo_name, lbls, best_k)


# %%
"""
Compare the results of KMeans, Spectral Clustering, and GMM by Gap statistic and silhouette score plots
"""
def wcss(X, labels):
    """Within‑cluster sum of squares (Euclidean)."""
    w = 0.0
    for c in np.unique(labels):
        Xi = X[labels == c]
        if Xi.shape[0] == 0:  # empty cluster
            continue
        ctr = Xi.mean(axis=0)
        w += ((Xi - ctr) ** 2).sum()
    return w


# -------------------------------------------------
# helper: Gap statistic for ONE (algo,k) pair
# -------------------------------------------------
def gap_single(X, k, w_obs, B=20, seed=42):
    rng = np.random.RandomState(seed)
    lows = X.min(axis=0);
    highs = X.max(axis=0)
    w_ref = []
    for b in range(B):
        Xb = rng.uniform(low=lows, high=highs, size=X.shape)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed + b)
        lbl_b = km.fit_predict(Xb)
        w_ref.append(wcss(Xb, lbl_b))
    w_ref = np.array(w_ref)
    gap = np.log(w_ref).mean() - np.log(w_obs)
    s_k = np.sqrt(((np.log(w_ref) - np.log(w_ref).mean()) ** 2).mean())
    return gap, s_k * np.sqrt(1 + 1 / B)


# -------------------------------------------------
# main grid
# -------------------------------------------------
algorithms = {
    "kmeans": lambda k: KMeans(n_clusters=k, n_init=50, max_iter=500, random_state=42),
    "spectral": lambda k: SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                             n_neighbors=15, assign_labels="kmeans", random_state=42),
    "gmm": lambda k: GaussianMixture(n_components=k, covariance_type="full",
                                     n_init=10, max_iter=500, random_state=42)
}

results = []  # collect dicts, then convert to DataFrame

print(f"Scanning k = {list(k_range)} for {list(algorithms.keys())} …")
for algo_name, ctor in algorithms.items():
    for k in k_range:
        # fit / predict
        model = ctor(k)
        labels = model.fit_predict(X_embed) if algo_name == "gmm" else model.fit(X_embed).labels_

        # metrics that need ≥2 clusters
        if len(np.unique(labels)) < 2:
            silhouette = np.nan;
            gap = np.nan;
            gap_se = np.nan
        else:
            silhouette = silhouette_score(X_embed, labels)

            w_k = wcss(X_embed, labels)  # dispersion for elbow + Gap
            gap, gap_se = gap_single(X_embed, k, w_k, B=20)  # 20 bootstraps

        # inertia‑like value for elbow: W_k (not kmeans inertia!)
        elbow = wcss(X_embed, labels)

        # BIC for GMM only
        bic = model.bic(X_embed) if algo_name == "gmm" else np.nan

        results.append(dict(
            algo=algo_name, k=k,
            silhouette=silhouette,
            elbow=elbow,
            gap=gap,
            gap_se=gap_se,
            bic=bic
        ))

metric_df = pd.DataFrame(results)

def gap_best_k(df_algo):
    """
    Tibshirani rule: smallest k with
        Gap(k) >= Gap(k+1) - gap_se(k+1)
    If that never happens, return k with largest Gap.
    """
    ks = sorted(df_algo["k"])
    gaps   = df_algo.set_index("k")["gap"].to_dict()
    gap_se = df_algo.set_index("k")["gap_se"].to_dict()

    for i, k in enumerate(ks[:-1]):          # up to penultimate
        next_k = ks[i+1]
        if np.isfinite(gaps[k]) and np.isfinite(gaps[next_k]):
            if gaps[k] >= gaps[next_k] - gap_se.get(next_k, 0):
                return k
    # fallback: k with max Gap
    return max(gaps, key=gaps.get)

print("\n======== Best k suggested by Gap statistic ========")
for algo, grp in metric_df.groupby("algo"):
    best_k_gap = gap_best_k(grp)
    print(f"{algo:<9} →  k = {best_k_gap}   (Gap = {grp.set_index('k')['gap'][best_k_gap]:0.3f})")

# -------------------------------------------------
# PLOTS  (one row per metric, three coloured lines)
# -------------------------------------------------
palette = dict(kmeans="#1f77b4", spectral="#ff7f0e", gmm="#2ca02c")

plt.figure(figsize=(6, 14))

# 1. silhouette
plt.subplot(4, 1, 1)
for algo, grp in metric_df.groupby("algo"):
    sns.lineplot(data=grp, x="k", y="silhouette", label=algo,
                 marker="o", color=palette[algo])
plt.title("Silhouette vs k")
plt.xlabel("k")
plt.ylabel("avg silhouette")
plt.grid(True)
plt.xticks(k_range)

# 2. elbow  (WCSS)
plt.subplot(4, 1, 2)
for algo, grp in metric_df.groupby("algo"):
    sns.lineplot(data=grp, x="k", y="elbow", label=algo,
                 marker="o", color=palette[algo])
plt.title("Elbow (within cluster SS)")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.grid(True)
plt.xticks(k_range)

# 3. Gap statistic
plt.subplot(4, 1, 3)
for algo, grp in metric_df.groupby("algo"):
    sns.lineplot(data=grp, x="k", y="gap", label=algo,
                 marker="o", color=palette[algo])
plt.title("Gap statistic")
plt.xlabel("k")
plt.ylabel("Gap")
plt.grid(True)
plt.xticks(k_range)

# 4. BIC  (GMM only)
plt.subplot(4, 1, 4)
sns.lineplot(data=metric_df[metric_df["algo"] == "gmm"],
             x="k", y="bic", marker="o", color=palette["gmm"])
plt.title("BIC (Gaussian Mixture)")
plt.xlabel("k")
plt.ylabel("BIC (lower = better)")
plt.grid(True)
plt.xticks(k_range)

plt.tight_layout()
plt.savefig(SHAP_clustering_path + "clustering_metrics_comparison.png",
            dpi=300, bbox_inches="tight")
plt.show()

# %%
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

def cluster_palette(labels):
    """
    Return a dict {label: hex‑color} using the fixed colour cycle.
    Works for numeric or string labels.
    """
    uniq = np.unique(labels)              # preserves sort order
    cols = list(islice(cycle(BASE_COLORS), len(uniq)))
    return dict(zip(uniq.astype(str), cols))   # seaborn hue → string

# ks_to_compare = [2, 3, 4]            # edit freely
ks_to_compare = [5]            # edit freely

# master containers
all_metric_rows = []                 # one row per (algo,k)
all_labels      = {}                 # {(algo,k): labels}

# ----------------------------------------------------------------------
# 1.  Fit every algorithm for every k, collect metrics
# ----------------------------------------------------------------------
for k in ks_to_compare:
    algo_dict = {
        "kmeans": KMeans(
            n_clusters=k, init="k-means++",
            n_init=50, max_iter=500, random_state=42),
        "spectral": SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors",
            n_neighbors=15, assign_labels="kmeans",
            random_state=42),
        "gmm": GaussianMixture(
            n_components=k, covariance_type="full",
            n_init=10, max_iter=500, random_state=42)
    }

    print(f"\n=== k = {k} =================================================")
    for name, model in algo_dict.items():
        # fit & label
        labels = (model.fit_predict(X_embed)
                  if name == "gmm" else model.fit(X_embed).labels_)
        all_labels[(name, k)] = labels

        # metrics
        if len(np.unique(labels)) < 2:
            sil = chs = db = np.nan
            print(f"{name:9s}: only one cluster – metrics skipped.")
        else:
            sil = silhouette_score(X_embed, labels)
            chs = calinski_harabasz_score(X_embed, labels)
            db  = davies_bouldin_score(X_embed, labels)

        bic = model.bic(X_embed) if name == "gmm" else np.nan

        all_metric_rows.append(dict(
            Algo=name, k=k,
            Silhouette=sil,
            CalinskiHarabasz=chs,
            DaviesBouldin=db,
            BIC=bic
        ))
        print(f"{name:9s}: Sil={sil:5.3f}  CH={chs:7.1f}  DB={db:5.3f}  "
              f"BIC={bic:8.1f}" if name=="gmm" else "")

# tidy DataFrame
metric_df = (pd.DataFrame(all_metric_rows)
             .set_index(["Algo", "k"])
             .sort_index())
print("\n=================== summary table ===================")
print(metric_df)

# save for later
metric_path = SHAP_clustering_path + "clustering_metrics_k2-3-4.csv"
metric_df.to_csv(metric_path)
print(f"\nMetric table saved → {metric_path}")

# ----------------------------------------------------------------------
# 2.  Visualise each (algo, k) label set with UMAP & feature plots
# ----------------------------------------------------------------------
param_grid = [                           # UMAP parameter combos
    {"metric": "euclidean",  "n_neighbors": 40, "min_dist": 0.1},
]

umap_dir = SHAP_clustering_path + "umap/"
os.makedirs(umap_dir, exist_ok=True)

def make_umap(lbls, algo, k, params):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
        metric=params["metric"],
        random_state=42
    )
    emb = reducer.fit_transform(X_embed)
    df = pd.DataFrame({
        "UMAP‑1": emb[:, 0], "UMAP‑2": emb[:, 1],
        "Cluster": lbls.astype(str)
    })

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="UMAP‑1", y="UMAP‑2",
                    hue="Cluster", palette=cluster_palette(lbls),
                    s=50, alpha=0.9, linewidth=0)
    ttl = (f"UMAP – {algo} (k={k}) | "
           f"metric={params['metric']}, nn={params['n_neighbors']}, "
           f"md={params['min_dist']}")
    plt.title(ttl); plt.axis("off")
    plt.legend(title="Cluster", frameon=False,
               bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    fn = (f"{umap_dir}{algo}_k{k}_m{params['metric']}"
          f"_nn{params['n_neighbors']}_md{params['min_dist']}.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"saved {fn}")

for (algo, k), lbls in all_labels.items():
    # silhouette / cluster‑size print‑out
    if len(np.unique(lbls)) > 1:
        sil_vals = silhouette_samples(X_embed, lbls)
        print(f"\n[{algo}, k={k}]  Avg silhouette: "
              f"{sil_vals.mean():0.3f}")
        for c in np.unique(lbls):
            n = (lbls == c).sum()
            mu = sil_vals[lbls == c].mean()
            sd = sil_vals[lbls == c].std()
            print(f"  – Cluster {c}: N={n:3d} | μ={mu:0.3f} | σ={sd:0.3f}")

    # two feature vs SHAP scatter plots
    for feat in ["PSG_Sleep_Dur", "Self_Sleep_Dur", "PSG_Sleep_Eff", "Self_Sleep_Eff", "Age_at_Scan"]:
        if feat not in df_SHIP.columns or feat not in shap_df.columns:
            continue
        dfp = pd.DataFrame({
            "Original": df_SHIP[feat].values,
            "SHAP": shap_df[feat].values,
            "Cluster": lbls.astype(str)
        })
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=dfp, x="Original", y="SHAP",
                        hue="Cluster", palette=cluster_palette(lbls),
                        s=50, linewidth=0)
        plt.xlabel(f"{feat} (Original)"); plt.ylabel("SHAP value")
        plt.title(f"{feat}: original vs SHAP – {algo}, k={k}")
        plt.legend(title="Cluster", frameon=False,
                   bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fp = (f"{SHAP_clustering_path}{algo}_k{k}_{feat}.png")
        plt.savefig(fp, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"saved {fp}")

    # UMAPs
    for p in param_grid:
        make_umap(lbls, algo, k, p)

# %%
sns.set_theme(style="ticks", context="paper")

k_final = 5
spec = SpectralClustering(
    n_clusters=k_final,
    affinity="nearest_neighbors",
    n_neighbors=15,
    assign_labels="kmeans",
    random_state=42
)
labels_final = spec.fit(X_embed).labels_.astype(str)

# 1. UMAP embedding with the agreed parameters
reducer = umap.UMAP(
    n_components=2, n_neighbors=40, min_dist=0.1,
    metric="euclidean", random_state=42
)
emb = reducer.fit_transform(X_embed)
df_umap = pd.DataFrame({"U1": emb[:,0], "U2": emb[:,1],
                        "Cluster": labels_final})

plt.figure(figsize=(7,5))
sns.scatterplot(data=df_umap, x="U1", y="U2",
                hue="Cluster",
                palette=cluster_palette(labels_final),
                s=50, alpha=0.9, linewidth=0)
plt.title(f"UMAP – Spectral clustering (k={k_final})")
# plt.axis("off")
plt.legend(title="Cluster", frameon=False,
           bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
umap_path = f"{SHAP_clustering_path}spectral_final_k{k_final}_umap.png"
plt.savefig(umap_path, dpi=300, bbox_inches="tight")
plt.show(); plt.close()
print(f"saved {umap_path}")

# 2. Feature‑vs‑SHAP scatter plots
scatter_dir = SHAP_clustering_path + "spectral_final_scatter/"
os.makedirs(scatter_dir, exist_ok=True)

for feat in ["Age_at_Scan", "PSG_Sleep_Dur", "Self_Sleep_Dur",
             "PSG_Sleep_Eff", "Self_Sleep_Eff"]:
    if feat not in df_SHIP.columns or feat not in shap_df.columns:
        continue
    dfp = pd.DataFrame({
        "Original": df_SHIP[feat].values,
        "SHAP":    shap_df[feat].values,
        "Cluster": labels_final
    })
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=dfp, x="Original", y="SHAP",
                    hue="Cluster",
                    palette=cluster_palette(labels_final),
                    s=50, linewidth=0)
    plt.xlabel(f"{feat} (Original)")
    plt.ylabel("SHAP value")
    plt.title(f"{feat}: Original vs SHAP – Spectral (k={k_final})")
    plt.legend(title="Cluster", frameon=False,
               bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    out = f"{scatter_dir}{feat}_spectral_k{k_final}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
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

sns.set_theme(style="ticks", context="paper")

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

    if feat not in shap_df.columns or feat not in df_SHIP.columns:
        ax.set_visible(False)
        continue

    dfp = pd.DataFrame({
        "Orig":  df_SHIP[feat].values,
        "SHAP":  shap_df[feat].values,
        "Cluster": labels_final
    })
    sns.scatterplot(
        data=dfp, x="Orig", y="SHAP",
        hue="Cluster",
        palette=cluster_palette(labels_final),
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
plt.savefig(os.path.join(fig_dir, fig_name), dpi=300,
            bbox_inches="tight")
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
    plt.savefig(out, dpi=300, bbox_inches="tight")
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
plt.savefig(feat_umap_path, dpi=300, bbox_inches="tight")
plt.show(); plt.close()
print(f"saved {feat_umap_path}")

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

for cl, idx in rep_idx.items():  #  rep_idx or rep_idx_mag

    # -------------------------------
    # extract that subject's interaction matrix
    # ivs is [sample, i, j]  (assume symmetric, diagonal = main effect)
    # -----------------------------------------------------------------
    interaction_values = ivs[idx]

    fig = shapiq.network_plot(
        first_order_values = interaction_values.get_n_order_values(1),
        second_order_values= interaction_values.get_n_order_values(2),
        feature_names      = feature_names,
        draw_legend = False
    )

    out = f"{net_dir}cluster_{cl}_rep_network.png"
    # title
    plt.title(f"Cluster {cl} representative subject", fontsize=14)
    plt.tight_layout()
    # save figure
    plt.savefig(out, dpi=300, bbox_inches="tight")
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

for cl in label_set:
    idx = np.where(labels_final == cl)[0]       # subjects in this cluster
    if len(idx) == 0:
        continue

    # -------------------------------------------------------------
    # 1st‑ and 2nd‑order arrays for every subject, then mean
    # -------------------------------------------------------------
    first_list  = []
    second_list = []
    for s in idx:
        iv = ivs[s]
        first_list.append(iv.get_n_order_values(1))   # shape (F,)
        second_list.append(iv.get_n_order_values(2))    # shape (F,F)

    first_avg  = np.mean(first_list,  axis=0)
    second_avg = np.mean(second_list, axis=0)

    # -------------------------------------------------------------
    # network plot on the averaged orders
    # -------------------------------------------------------------
    fig = shapiq.network_plot(
        first_order_values  = first_avg,
        second_order_values = second_avg,
        feature_names       = feature_names,
        draw_legend         = False,
    )
    plt.title(f"Cluster {cl} – mean interaction network (n={len(idx)})",
              fontsize=13, pad=10)
    plt.tight_layout()

    out = f"{net_dir}cluster_{cl}_avg_network.png"
    # save figure
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()
    print(f"saved {out}")

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

first_avg  = np.mean(first_all,  axis=0)       # (F,)
second_avg = np.mean(second_all, axis=0)       # (F, F)

# ---------------------------------------------------------------------
# 2.  Plot the grand‑average network
# ---------------------------------------------------------------------
edge_max = np.max(np.abs(second_avg))          # symmetric colour scale

fig = shapiq.network_plot(
    first_order_values  = first_avg,
    second_order_values = second_avg,
    feature_names       = feature_names,             # same feature order
    draw_legend         = False,
)

plt.title("Mean 1st/2nd order interactions (all subjects)",
          fontsize=13, pad=10)
plt.tight_layout()

out_all = SHAP_clustering_path + "network_mean_all_subjects.png"
# save
plt.savefig(out_all, dpi=300, bbox_inches="tight")
plt.show(); plt.close()
print(f"saved {out_all}")
