import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import XGBoost_pipeline
from XGBoost_pipeline import AdMLPipeline
import xgboost as xgb

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
parser = argparse.ArgumentParser(description='SHAP-IQ for XGBoost, SHIP_Trend dataset.')
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

print(f"\nStarting SHAP-IQ for XGBoost pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/{target}/SHIP_Trend_htcondor/'

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

# load saved shap value by pickle
# print(f"\nLoading SHAP-IQ values for {feature_comb}, {target}.\n")
# ivs_saved_path = shapiq_path + 'ivs.pkl'
# with open(ivs_saved_path, 'rb') as f:
#     ivs = pickle.load(f)

# load saved shap value by pickle
print(f"\nLoading SHAP values for {feature_comb}, {target}.\n")
shap_values_saved_path = SHAP_path + 'explanation_tdep_manual_dmatrix.pkl'
with open(shap_values_saved_path, 'rb') as f:
    explanation_train_set_train = pickle.load(f)

explanation = explanation_train_set_train

# %%
# plot the beeswarm plot
shap.plots.beeswarm(explanation)

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

# ── helper to make *all three* figures for one algorithm ───────────────
def visualise_algo(algo_name, lbls):
    print(f"\nCreating visualisations for «{algo_name}»")

    param_grid = [
        # ――― a balanced starter set ―――
        {"metric": "euclidean", "n_neighbors": 15, "min_dist": 0.1},
        {"metric": "euclidean", "n_neighbors": 30, "min_dist": 0.1},
        {"metric": "manhattan", "n_neighbors": 30, "min_dist": 0.1},
        {"metric": "chebyshev", "n_neighbors": 30, "min_dist": 0.1},
        # add / remove rows as you like
    ]

    for p in param_grid:
        metric = p["metric"]
        nn = p["n_neighbors"]
        md = p["min_dist"]

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=md,
            metric=metric,
            random_state=42
        )
        emb = reducer.fit_transform(X_embed)

        df_umap = pd.DataFrame({
            "UMAP‑1": emb[:, 0],
            "UMAP‑2": emb[:, 1],
            "Cluster": lbls.astype(str)
        })

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=df_umap,
            x="UMAP‑1", y="UMAP‑2",
            hue="Cluster",
            palette="viridis",
            s=50, alpha=0.9, linewidth=0
        )

        title = (f"UMAP ({algo_name}, k={best_k})  |  "
                 f"metric={metric}, n_neighbors={nn}, min_dist={md}")
        plt.title(title)
        plt.axis("off")
        plt.legend(title="Cluster", frameon=False,
                   bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        umap_save_path = SHAP_clustering_path + "umap/"
        if not os.path.exists(umap_save_path):
            os.makedirs(umap_save_path)
        fname = umap_save_path + f"{algo_name}_k{best_k}_{metric}_nn{nn}_md{md}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"saved {fname}")

# ── run the helper for every algorithm ─────────────────────────────────
for algo_name, lbls in labels_dict.items():
    visualise_algo(algo_name, lbls)

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
ks_to_compare = [4]            # edit freely

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
    {"metric": "euclidean",  "n_neighbors": 15, "min_dist": 0.1},
    {"metric": "euclidean",  "n_neighbors": 30, "min_dist": 0.1},
    {"metric": "manhattan",  "n_neighbors": 30, "min_dist": 0.1},
    {"metric": "chebyshev",  "n_neighbors": 30, "min_dist": 0.1},
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
    # plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"saved {fn}")

# for (algo, k), lbls in all_labels.items():
#     # silhouette / cluster‑size print‑out
#     if len(np.unique(lbls)) > 1:
#         sil_vals = silhouette_samples(X_embed, lbls)
#         print(f"\n[{algo}, k={k}]  Avg silhouette: "
#               f"{sil_vals.mean():0.3f}")
#         for c in np.unique(lbls):
#             n = (lbls == c).sum()
#             mu = sil_vals[lbls == c].mean()
#             sd = sil_vals[lbls == c].std()
#             print(f"  – Cluster {c}: N={n:3d} | μ={mu:0.3f} | σ={sd:0.3f}")
#
#     # two feature vs SHAP scatter plots
#     for feat in ["PSG_Sleep_Dur", "Self_Sleep_Dur", "PSG_Sleep_Eff", "Self_Sleep_Eff", "Age_at_Scan"]:
#         if feat not in df_SHIP.columns or feat not in shap_df.columns:
#             continue
#         dfp = pd.DataFrame({
#             "Original": df_SHIP[feat].values,
#             "SHAP": shap_df[feat].values,
#             "Cluster": lbls.astype(str)
#         })
#         plt.figure(figsize=(7, 5))
#         sns.scatterplot(data=dfp, x="Original", y="SHAP",
#                         hue="Cluster", palette=cluster_palette(lbls),
#                         s=50, linewidth=0)
#         plt.xlabel(f"{feat} (Original)"); plt.ylabel("SHAP value")
#         plt.title(f"{feat}: original vs SHAP – {algo}, k={k}")
#         plt.legend(title="Cluster", frameon=False,
#                    bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         fp = (f"{SHAP_clustering_path}{algo}_k{k}_{feat}.png")
#         # plt.savefig(fp, dpi=300, bbox_inches="tight")
#         plt.show()
#         plt.close()
#         print(f"saved {fp}")

    # UMAPs
    for p in param_grid:
        make_umap(lbls, algo, k, p)

# %%
"""
Try HDBSCAN on the SHAP values
"""
def cluster_palette_no_noise(labels):
    clusters = [c for c in np.unique(labels) if c != -1]
    cols = list(islice(cycle(BASE_COLORS), len(clusters)))
    return dict(zip(map(str, clusters), cols))

# hdb_grid = [
#     {"min_cluster_size": 20, "min_samples": 8,  "csel_eps": 0.0},
#     {"min_cluster_size": 15, "min_samples": 6,  "csel_eps": 0.0},
#     {"min_cluster_size": 10, "min_samples": 5,  "csel_eps": 0.0},
#     {"min_cluster_size": 20, "min_samples": 8,  "csel_eps": 0.2},  # soft splitting
# ]
# hdb_grid = [
#     #  metric      min_cluster_size  min_samples  csel_eps
#     ("euclidean",  25,               15,          0.00),
#     ("euclidean",  25,               12,          0.10),
#     ("euclidean",  25,               12,          0.20),
#     ("cosine",     25,               15,          0.00),
#     ("cosine",     25,               15,          0.10),
#     ("cosine",     15,               10,          0.10),
# ]
metrics = ["euclidean", "cosine"]
mcs_list = [10, 25, 45]
eps_list = [0.00, 0.05, 0.10]

hdb_grid = []
for metric in metrics:
    for mcs in mcs_list:
        ms_options = [max(5, int(mcs*0.4)), max(5, int(np.log(len(X_embed))))]
        for ms in ms_options:
            for eps in eps_list:
                hdb_grid.append((metric, mcs, ms, eps))

# if feature scales differ wildly, apply StandardScaler here
X_use = X_embed            # your SHAP values already meaningful

# directories
hdb_dir   = SHAP_clustering_path + "hdbscan/"
umap_dir  = hdb_dir + "umap/"
os.makedirs(umap_dir, exist_ok=True)

metric_rows = []
labels_dict = {}

# ────────────────────────────────────────────────────────────────────
# 3.  Fit & score each parameter combo
# ────────────────────────────────────────────────────────────────────
# for p in hdb_grid:
#     mcs, ms, eps = p["min_cluster_size"], p["min_samples"], p["csel_eps"]
for metric, mcs, ms, eps in hdb_grid:
    tag = f"metric{metric}_mcs{mcs}_ms{ms}_eps{eps}"
    print(f"\n=== HDBSCAN {tag} ===")

    hdb = HDBSCAN(min_cluster_size=mcs,
                  min_samples=ms,
                  cluster_selection_epsilon=eps,
                  metric="euclidean",
                  cluster_selection_method="eom",
                  allow_single_cluster=False,
                  n_jobs=-1)
    lbls = hdb.fit_predict(X_use)
    labels_dict[tag] = lbls

    clusters = [c for c in np.unique(lbls) if c != -1]   # ignore noise
    if len(clusters) < 2:
        print("   <2 clusters – metrics undefined.")
        sil = chs = db = np.nan
    else:
        sil = silhouette_score(X_use, lbls)
        chs = calinski_harabasz_score(X_use, lbls)
        db  = davies_bouldin_score(X_use, lbls)
        print(f"   clusters={len(clusters):2d} | Sil={sil:0.3f} | "
              f"CH={chs:0.1f} | DB={db:0.3f}")

    metric_rows.append(dict(
        tag=tag, min_cluster_size=mcs, min_samples=ms,
        csel_eps=eps, n_clusters=len(clusters),
        Silhouette=sil, CalinskiHarabasz=chs, DaviesBouldin=db
    ))

# tidy summary
hdb_df = (pd.DataFrame(metric_rows)
            .set_index("tag")
            .sort_values("Silhouette", ascending=False))
print("\n=========== HDBSCAN summary ===========")
print(hdb_df)

# save
hdb_df.to_csv(hdb_dir + "hdbscan_metrics.csv")

# ────────────────────────────────────────────────────────────────────
# 4.  Visualisations (feature plots + UMAP) for every setting
# ────────────────────────────────────────────────────────────────────
umap_params = [{"metric":"euclidean","n_neighbors":30,"min_dist":0.1}]

for tag, lbls in labels_dict.items():
    # ----- silhouette breakdown -----
    if len(np.unique(lbls)) > 1:
        sil_vals = silhouette_samples(X_use, lbls)
        print(f"\n[{tag}] avg silhouette = {sil_vals.mean():0.3f}")
        for c in np.unique(lbls):
            n = (lbls == c).sum()
            mu = sil_vals[lbls==c].mean(); sd = sil_vals[lbls==c].std()
            print(f"  • Cluster {c}: N={n:3d} | μ={mu:0.3f} | σ={sd:0.3f}")

    # ----- feature‑vs‑SHAP scatter -----
    # for feat in ["PSG_Sleep_Dur","Self_Sleep_Dur",
    #              "PSG_Sleep_Eff","Self_Sleep_Eff","Age_at_Scan"]:
    for feat in ["Age_at_Scan"]:
        if feat not in df_SHIP.columns or feat not in shap_df.columns:
            continue
        dfp = pd.DataFrame({
            "Original": df_SHIP[feat].values,
            "SHAP":    shap_df[feat].values,
            "Cluster": lbls.astype(str)
        })

        dfp["Cluster"] = lbls.astype(str)
        df_noise = dfp[dfp["Cluster"] == "-1"]
        df_clusters = dfp[dfp["Cluster"] != "-1"]

        plt.figure(figsize=(7,5))
        sns.scatterplot(data=df_clusters, x="Original", y="SHAP",
                        hue="Cluster", palette=cluster_palette_no_noise(lbls),
                        s=50, linewidth=0, legend="brief")

        # noise points as black ×
        if not df_noise.empty:
            sns.scatterplot(data=df_noise, x="Original", y="SHAP",
                            marker="x", color="k", s=60, linewidth=1,
                            label="Noise", legend="brief")
        plt.xlabel(f"{feat} (Original)"); plt.ylabel("SHAP value")
        plt.title(f"{feat}: original vs SHAP – HDBSCAN {tag}")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05,1),
                   loc="upper left", frameon=False)
        plt.tight_layout(); plt.show(); plt.close()

    # ----- UMAP -----
    for up in umap_params:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=up["n_neighbors"],
            min_dist=up["min_dist"],
            metric=up["metric"],
            random_state=42
        )
        emb = reducer.fit_transform(X_use)
        dfu = pd.DataFrame({"U1":emb[:,0],"U2":emb[:,1],
                            "Cluster":lbls.astype(str)})

        dfu["Cluster"] = lbls.astype(str)
        df_noise = dfu[dfu["Cluster"] == "-1"]
        df_clusters = dfu[dfu["Cluster"] != "-1"]

        plt.figure(figsize=(7,5))
        sns.scatterplot(data=df_clusters, x="U1", y="U2",
                        hue="Cluster", palette=cluster_palette_no_noise(lbls),
                        s=50, alpha=0.9, linewidth=0, legend=False)

        if not df_noise.empty:
            sns.scatterplot(data=df_noise, x="U1", y="U2",
                            marker="x", color="k", s=60, linewidth=1,
                            label="Noise", legend=False)
        plt.title(f"UMAP – HDBSCAN {tag}")
        plt.axis("off")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05,1),
                   loc="upper left", frameon=False)
        plt.tight_layout(); plt.show(); plt.close()

# %%
"""
Find better u-map shape
"""
k_gmm = 4                                   # ← adjust if you prefer
gmm   = GaussianMixture(n_components=k_gmm,
                        covariance_type="full",
                        n_init=10,
                        max_iter=500,
                        random_state=42
                        )
lbls  = gmm.fit_predict(X_embed)            # 1‑D int array
lbls  = lbls.astype(str)                    # make it str for hue

metrics     = ["euclidean", "manhattan", "chebyshev"]  # cosine is bad
n_neighbors = [10, 20, 40]                 # local ⇆ global balance
min_dists   = [0.0, 0.1, 0.3]              # cluster tightness

umap_grid = [
    {"metric": m, "n_neighbors": nn, "min_dist": md}
    for m in metrics
    for nn in n_neighbors
    for md in min_dists
]

def umap_score(embedding, labels):
    # ignore noise (‑1) so it doesn’t drag score down
    mask = labels != "-1"
    if mask.sum() < 3 or len(np.unique(labels[mask])) < 2:
        return -1   # invalid / too few points
    return silhouette_score(embedding[mask], labels[mask].astype(int))

best_runs = []          # collect (score, params, emb)

for prm in umap_grid:
    reducer = umap.UMAP(
        n_components=2, random_state=42, **prm
    )
    emb = reducer.fit_transform(X_embed)
    score = umap_score(emb, lbls.astype(str))
    best_runs.append((score, prm, emb))

# keep the top‑3 settings by score
best_runs.sort(reverse=True, key=lambda x: x[0])
top_runs = best_runs[:3]
print("\nBest UMAP settings:")
for s, prm, _ in top_runs:
    print(f"  score={s:0.3f}  |  {prm}")

for rank, (s, prm, emb) in enumerate(top_runs, 1):
    df = pd.DataFrame({
        "U1": emb[:,0], "U2": emb[:,1],
        "Cluster": lbls.astype(str)
    })

    plt.figure(figsize=(7,5))
    # clusters
    sns.scatterplot(data=df[df.Cluster!="-1"], x="U1", y="U2",
                    hue="Cluster",
                    palette=cluster_palette(df[df.Cluster!="-1"].Cluster),
                    s=50, linewidth=0, alpha=0.9, legend=False)
    # noise
    sns.scatterplot(data=df[df.Cluster=="-1"], x="U1", y="U2",
                    marker="x", color="k", s=60, linewidth=1, legend=False)

    title = (f"UMAP Top {rank}  (score={s:0.3f})  |  "
             f"metric={prm['metric']}, nn={prm['n_neighbors']}, "
             f"md={prm['min_dist']}")
    plt.title(title); plt.axis("off"); plt.tight_layout()
    plt.show()

# %%
# "spectral": SpectralClustering(
#     n_clusters=best_k,
#     affinity="nearest_neighbors",  # nearest‑neighbor graph works well for medium‑large n
#     n_neighbors=15,                # tweak if your data are very dense/sparse
#     random_state=42,
#     assign_labels="kmeans"         # gives crisp labels
# ),
k_spectral = 5                                   # ← adjust if you prefer
SC = SpectralClustering(
    n_clusters=k_spectral,
    affinity="nearest_neighbors",
    n_neighbors=15,
    random_state=42
)
lbls  = SC.fit_predict(X_embed)            # 1‑D int array
lbls  = lbls.astype(str)                    # make it str for hue

# ------------------------------------------------------------------
# 1.  Loop through the whole grid
# ------------------------------------------------------------------
for idx, prm in enumerate(umap_grid, 1):

    # ---- fit UMAP --------------------------------------------------
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=prm["n_neighbors"],
        min_dist=prm["min_dist"],
        metric=prm["metric"],
        random_state=42
    )
    emb = reducer.fit_transform(X_embed)

    # ---- build plotting DataFrame ---------------------------------
    df = pd.DataFrame({
        "U1": emb[:, 0],
        "U2": emb[:, 1],
        "Cluster": lbls            # lbls already str from GMM
    })

    # ---- draw scatter ---------------------------------------------
    plt.figure(figsize=(7, 5))

    # clusters (filled circles)
    sns.scatterplot(
        data=df[df.Cluster != "-1"], x="U1", y="U2",
        hue="Cluster",
        palette=cluster_palette(df[df.Cluster != "-1"].Cluster),
        s=50, linewidth=0, alpha=0.9, legend=False
    )

    # noise points (black ×)
    sns.scatterplot(
        data=df[df.Cluster == "-1"], x="U1", y="U2",
        marker="x", color="k", s=60, linewidth=1, legend=False
    )

    title = (f"UMAP {idx}/{len(umap_grid)} | "
             f"metric={prm['metric']}, nn={prm['n_neighbors']}, "
             f"md={prm['min_dist']}")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # ---- save &/or show -------------------------------------------
    # fname = (f"{umap_dir}umap_{idx:02d}_m-{prm['metric']}"
    #          f"_nn-{prm['n_neighbors']}_md-{prm['min_dist']}.png")
    # plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    # print(f"saved {fname}")

# %%
for feat in ["PSG_Sleep_Dur","Self_Sleep_Dur", "PSG_Sleep_Eff","Self_Sleep_Eff","Age_at_Scan"]:
    if feat not in df_SHIP.columns or feat not in shap_df.columns:
        print(f"Feature '{feat}' not found – skipping.")
        continue

    df_plot = pd.DataFrame({
        "Original": df_SHIP[feat].values,
        "SHAP":    shap_df[feat].values,
        "Cluster": lbls.astype(str)
    })

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_plot, x="Original", y="SHAP",
                    hue="Cluster", palette=cluster_palette(df[df.Cluster != "-1"].Cluster), s=50, linewidth=0)
    plt.xlabel(f"{feat} (Original)"); plt.ylabel(f"SHAP value")
    plt.title(f"{feat}: Original vs. SHAP  (gmm, k=4)")
    plt.legend(title="Cluster", frameon=False,
               bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    # fname = f"{SHAP_clustering_path}{algo_name}_{feat}.png"
    # plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()
    # print(f"  → scatter figure saved: {fname}")

# %%
"""
Color umap by SHAP value
"""
from matplotlib.colors import LinearSegmentedColormap

# 1. build a 3‑colour list  (blue, gray, red)
GRAD_COLORS = ["#053061",  "#f7f7f7",  "#b2182b"]   # ← tweak if needed
gray_cmap   = LinearSegmentedColormap.from_list("BlueGrayRed", GRAD_COLORS, N=256)

prm = {"metric": "euclidean", "n_neighbors": 20, "min_dist": 0.1}
reducer = umap.UMAP(n_components=2, random_state=42, **prm)
embedding = reducer.fit_transform(X_embed)        # shape (n_samples, 2)

# put coords into a frame so we can re‑use it
base_df = pd.DataFrame({"U1": embedding[:, 0], "U2": embedding[:, 1]})

# directory for figures
# umap_feat_dir = SHAP_clustering_path + "umap_by_feature/"
# os.makedirs(umap_feat_dir, exist_ok=True)

# ------------------------------------------------------------------
# list the SHAP features you want to visualise
# ------------------------------------------------------------------
features_to_colour = [
    "PSG_Sleep_Dur", "Self_Sleep_Dur",
    "PSG_Sleep_Eff", "Self_Sleep_Eff",
    "Age_at_Scan"
]

CMAP  = "RdBu_r"          # blue‑white‑red, reversed
VMIN, VMAX = -4, 4        # fixed range for *all* plots

for feat in features_to_colour:
    if feat not in shap_df.columns:
        print(f"{feat} missing – skipped."); continue

    vals = shap_df[feat].values.clip(VMIN, VMAX)  # squash extreme outliers
    df   = base_df.copy()                         # base_df has U1,U2 columns
    df["SHAP"] = vals

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(df["U1"], df["U2"],
                     c=vals, cmap='coolwarm',  # CMAP
                     vmin=VMIN, vmax=VMAX,
                     s=50, linewidths=0)

    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_label(f"SHAP value ({feat})", rotation=270, labelpad=15)
    cbar.set_ticks([VMIN, 0, VMAX])

    plt.title(f"UMAP coloured by {feat} SHAP (common scale)")
    plt.axis("off")
    plt.tight_layout()

    # fn = f"{umap_feat_dir}umap_BY_{feat}.png"
    # plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()
    # print(f"saved {fn}")