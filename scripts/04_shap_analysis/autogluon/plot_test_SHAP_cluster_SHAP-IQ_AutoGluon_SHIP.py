import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import AutoGluon_pipeline
from AutoGluon_pipeline import AdMLPipeline
from autogluon.tabular import TabularPredictor

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import shapiq
import pickle
from joblib import load
from datetime import datetime

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

# load saved shap value by pickle
print(f"\nLoading SHAP-IQ values for {feature_comb}, {target}.\n")
ivs_saved_path = shapiq_path + 'ivs.pkl'
with open(ivs_saved_path, 'rb') as f:
    ivs = pickle.load(f)

# load saved shap value by pickle
print(f"\nLoading SHAP values for {feature_comb}, {target}.\n")
shap_values_saved_path = SHAP_path + 'explanation_train_set_train.pkl'
with open(shap_values_saved_path, 'rb') as f:
    explanation_train_set_train = pickle.load(f)

explanation = explanation_train_set_train

# %%
# output_of_force_plot = shap.force_plot(explanation, matplotlib=False, show=False)
# file = SHAP_path + 'force_plot_average.html'
# shap.save_html(file, output_of_force_plot)

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
Try HDBSCAN clustering, not suitable, I need algorithm that can label all points
"""
# # from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline      import Pipeline
# from sklearn.metrics       import silhouette_score, silhouette_samples
# from sklearn.cluster       import HDBSCAN           # native scikit-learn
# # If you also want DBSCAN for comparison:
# # from sklearn.cluster import DBSCAN
#
# # --- data --------------------------------------------------------------
# # X_embed = shap_df.values        # (subjects × 8 SHAP features)
# feat = 'PSG_Sleep_Dur'
#
# # 2-D array (subjects × 2)
# X_embed = np.column_stack([
#     df_SHIP[feat].values,      # real value
#     shap_df[feat].values       # SHAP value
# ])
#
# # --- pipeline ----------------------------------------------------------
# pipeline = Pipeline(
#     steps=[
#         # ('scale',   StandardScaler()),
#         ('cluster', HDBSCAN(
#             min_cluster_size          = 15,
#             # min_samples               = 15,
#             # metric                    = 'euclidean',
#             cluster_selection_method  = 'leaf',
#             cluster_selection_epsilon=0.1
#             # n_jobs                    = -1
#         ))
#     ],
#     verbose=True
# )
#
# # --- fit ---------------------------------------------------------------
# pipeline.fit(X_embed)
# labels = pipeline.named_steps['cluster'].labels_
#
# # --- evaluation --------------------------------------------------------
# mask   = labels != -1
# k      = len(set(labels)) - (1 if -1 in labels else 0)
# print(f'Clusters: {k} | Noise points: {(~mask).sum()}')
#
# if k >= 2:
#     sil_global = silhouette_score(X_embed[mask], labels[mask])
#     sil_indiv  = silhouette_samples(X_embed[mask], labels[mask])
#     print(f'Average silhouette (ex-noise): {sil_global:0.3f}')
#     for lab in np.unique(labels[mask]):
#         vals = sil_indiv[labels[mask] == lab]
#         print(f'  – Cluster {lab}: N={vals.size:3d} | μ={vals.mean():.3f} | σ={vals.std():.3f}')
# else:
#     print('Silhouette not defined because fewer than two non-noise clusters were found.')
#
# feature = 'PSG_Sleep_Dur'
#
# x = df_SHIP[feature].values
# y = shap_df[feature].values
# clusters = pd.Series(labels, name='cluster')
#
# plt.figure(figsize=(7,5))
# for c in sorted(clusters.unique()):
#     idx = clusters == c
#     plt.scatter(x[idx], y[idx],
#                 marker='x' if c == -1 else 'o',
#                 s=30,
#                 label='Noise' if c == -1 else f'Cluster {c}')
# plt.xlabel(feature)
# plt.ylabel('SHAP value')
# plt.title('PSG_Sleep_Dur vs. SHAP value\n(coloured by HDBSCAN clusters)')
# plt.legend(frameon=False)
# plt.tight_layout()
# plt.show()

# %%
"""
Try K-means clustering
"""
from sklearn.cluster      import KMeans
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import silhouette_score, silhouette_samples

# ----------------------------------------------------------------------
# 1. Build the 2-D embedding (same as before)
X_embed = shap_df.values
feat = 'PSG_Sleep_Dur'
# X_embed = np.column_stack([
#     df_SHIP[feat].values,      # raw value
#     shap_df[feat].values       # SHAP value
# ])

# ----------------------------------------------------------------------
# 2.  QUICK grid-search to pick a good k  (3 ↔ 4 as you requested)
silhouette_by_k = {}
for k in (3, 4):
    pipe = Pipeline([
        ('kmeans',  KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=50,          # robust to bad initialisations
            random_state=42
        ))
    ])
    labels_k = pipe.fit_predict(X_embed)
    sil      = silhouette_score(X_embed, labels_k)
    silhouette_by_k[k] = sil
    print(f'k = {k} → avg silhouette = {sil:0.3f}')

best_k = max(silhouette_by_k, key=silhouette_by_k.get)
print(f'\nChosen k = {best_k} (highest silhouette)')

# ----------------------------------------------------------------------
# 3.  Final pipeline with the chosen k
pipeline = Pipeline([
    ('kmeans',  KMeans(
        n_clusters        = best_k,  # best_k
        init              = 'k-means++',
        n_init            = 50,
        random_state      = 42,
        verbose           = 0
    ))
])
labels = pipeline.fit_predict(X_embed)

# ----------------------------------------------------------------------
# 4.  Detailed silhouette per cluster
sil_global = silhouette_score(X_embed, labels)
sil_indiv  = silhouette_samples(X_embed, labels)

print(f'\nAverage silhouette: {sil_global:0.3f}')
for lab in range(best_k):  # best_k
    vals = sil_indiv[labels == lab]
    print(f'  – Cluster {lab}: N={vals.size:3d} | μ={vals.mean():0.3f} | σ={vals.std():0.3f}')

# ----------------------------------------------------------------------
# 5.  Scatter plot (unchanged apart from label set)
import matplotlib.pyplot as plt

x = df_SHIP[feat].values
y = shap_df[feat].values
clusters = pd.Series(labels, name='cluster')

plt.figure(figsize=(7,5))
for c in sorted(clusters.unique()):
    idx = clusters == c
    plt.scatter(x[idx], y[idx],
                s=30,
                label=f'Cluster {c}')
plt.xlabel(feat)
plt.ylabel('SHAP value')
plt.title(f'{feat}: raw vs. SHAP (K-means, k={best_k})')  # best_k
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

feat = 'Age_at_Scan'
x = df_SHIP[feat].values
y = shap_df[feat].values
clusters = pd.Series(labels, name='cluster')

plt.figure(figsize=(7,5))
for c in sorted(clusters.unique()):
    idx = clusters == c
    plt.scatter(x[idx], y[idx],
                s=30,
                label=f'Cluster {c}')
plt.xlabel(feat)
plt.ylabel('SHAP value')
plt.title(f'{feat}: raw vs. SHAP (K-means, k={best_k})')  # best_k
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
"""
A more comprehensive k-means grid-search
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------
# 1. Define the data for clustering (SHAP values)
# Using all SHAP features for clustering
X_embed = shap_df.values

# ----------------------------------------------------------------------
# 2. Grid-search for k over the recommended range (2 to 15)
k_range = range(2, 16) # Test k from 2 up to 15
silhouette_by_k = {}
inertia_by_k = {} # Dictionary to store inertia for Elbow Method

print("Calculating Silhouette scores for k in", list(k_range)) # Show the list
for k in k_range:
    # *** FIX: Correctly initialize Pipeline with steps ***
    pipe = Pipeline([
        # ('scale', StandardScaler()),
        ('kmeans', KMeans(
            n_clusters=k,
            n_init=10, # Lower n_init for faster grid search
            random_state=42,  # 42
        ))
    ])
    # labels_k = pipe.fit_predict(X_embed)
    kmeans_model = pipe.fit(X_embed)  # Fit the pipeline
    labels_k = kmeans_model.predict(X_embed)  # Predict labels
    inertia = kmeans_model.named_steps['kmeans'].inertia_
    inertia_by_k[k] = inertia
    print(f'k = {k} → Inertia = {inertia:.2f}', end='')  # Print inertia first

    # Ensure there's more than 1 cluster label before calculating silhouette score
    if len(set(labels_k)) > 1:
        sil = silhouette_score(X_embed, labels_k)

        # X_scaled_for_eval = kmeans_model.named_steps['scale'].transform(X_embed)
        # # Calculate silhouette score using the scaled data and the labels
        # sil = silhouette_score(X_scaled_for_eval, labels_k)
        silhouette_by_k[k] = sil
        print(f'k = {k} → avg silhouette = {sil:0.3f}')
    else:
        print(f'k = {k} → Only 1 cluster found, cannot calculate silhouette score.')
        silhouette_by_k[k] = -1 # Assign a low score or handle as needed

# ----------------------------------------------------------------------
# 2b. Plot Silhouette scores vs. k using Seaborn
# plt.figure(figsize=(8, 5))
# *** Use seaborn.lineplot ***
# sns.lineplot(x=list(silhouette_by_k.keys()), y=list(silhouette_by_k.values()), marker='o')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Average Silhouette Score")
# plt.title("Silhouette Score vs. Number of Clusters (k)")
# plt.xticks(list(k_range)) # Ensure all k values are shown as ticks
# plt.grid(True)
# plt.show()
plt.figure(figsize=(5, 7)) # Make figure wider for two subplots
# Plot Silhouette Score
plt.subplot(2, 1, 1) # 1 row, 2 columns, 1st subplot
sns.lineplot(x=list(silhouette_by_k.keys()), y=list(silhouette_by_k.values()), marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.title("Silhouette Score vs. k")
plt.xticks(list(k_range)) # Ensure all k values are shown as ticks
plt.grid(True)

# Plot Elbow Method (Inertia)
plt.subplot(2, 1, 2) # 1 row, 2 columns, 2nd subplot
sns.lineplot(x=list(inertia_by_k.keys()), y=list(inertia_by_k.values()), marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method (Inertia vs. k)")
plt.xticks(list(k_range)) # Ensure all k values are shown as ticks
plt.grid(True)

plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.show()

# ----------------------------------------------------------------------
# 3. Select best k and run final pipeline
if not silhouette_by_k or all(score == -1 for score in silhouette_by_k.values()):
    print("\nError: No valid silhouette scores calculated. Cannot determine best k.")
    # Handle error case, e.g., exit or set a default k
    best_k = 3 # Defaulting to 3 as an example
    print(f"Warning: Proceeding with default k = {best_k}")
else:
    # Find the k that yielded the highest silhouette score
    best_k = max(silhouette_by_k, key=silhouette_by_k.get)
    print(f'\nChosen k = {best_k} (highest silhouette score: {silhouette_by_k[best_k]:.3f})')

# Final pipeline with the chosen k
pipeline = Pipeline([
    # ('scale',  StandardScaler()),
    ('kmeans', KMeans(
        n_clusters=best_k,
        init='k-means++',
        n_init=50, # Use higher n_init for the final model
        random_state=42,  # 42
        verbose=0
    ))
])
labels = pipeline.fit_predict(X_embed)

# ----------------------------------------------------------------------
# 4. Detailed silhouette per cluster for the chosen k
# Check if clustering resulted in more than one label
if len(np.unique(labels)) > 1:
    sil_global = silhouette_score(X_embed, labels)
    sil_indiv = silhouette_samples(X_embed, labels)

    print(f'\nAverage silhouette for k={best_k}: {sil_global:0.3f}')
    # Use np.unique to handle cases where cluster labels might not be contiguous
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        # Filter out potential noise points if any algorithm other than K-Means were used
        if lab == -1: continue
        vals = sil_indiv[labels == lab]
        # Check if cluster is not empty before calculating mean/std
        if vals.size > 0:
             print(f'  – Cluster {lab}: N={vals.size:3d} | μ={vals.mean():0.3f} | σ={vals.std():0.3f}')
        else:
             print(f'  – Cluster {lab}: N=0')
else:
    print(f'\nClustering with k={best_k} resulted in only one cluster or failed. Cannot calculate detailed silhouette scores.')

# ----------------------------------------------------------------------
# 5. Scatter plots for specific features (using the final labels from best_k)
#    (Using seaborn for scatter plots as well for consistency)

# Plot for 'PSG_Sleep_Dur'
feat1 = 'PSG_Sleep_Dur'
if feat1 in df_SHIP.columns and feat1 in shap_df.columns:
    plot_df1 = pd.DataFrame({
        'Original Value': df_SHIP[feat1].values,
        'SHAP Value': shap_df[feat1].values,
        'Cluster': labels.astype(str) # Convert labels to string for categorical plotting
    })

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df1, x='Original Value', y='SHAP Value', hue='Cluster', palette='viridis', s=50)
    plt.xlabel(feat1 + " (Original Value)")
    plt.ylabel('SHAP value for ' + feat1)
    plt.title(f'{feat1}: Original vs. SHAP (K-means, k={best_k})')
    plt.legend(title='Cluster', frameon=False)
    plt.tight_layout()
    plt.show()
else:
    print(f"Feature '{feat1}' not found in both DataFrames.")

# Plot for 'Age_at_Scan'
feat2 = 'Age_at_Scan'
if feat2 in df_SHIP.columns and feat2 in shap_df.columns:
    plot_df2 = pd.DataFrame({
        'Original Value': df_SHIP[feat2].values,
        'SHAP Value': shap_df[feat2].values,
        'Cluster': labels.astype(str) # Convert labels to string
    })

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df2, x='Original Value', y='SHAP Value', hue='Cluster', palette='viridis', s=50)
    plt.xlabel(feat2 + " (Original Value)")
    plt.ylabel('SHAP value for ' + feat2)
    plt.title(f'{feat2}: Original vs. SHAP (K-means, k={best_k})')
    plt.legend(title='Cluster', frameon=False)
    plt.tight_layout()
    plt.show()
else:
    print(f"Feature '{feat2}' not found in both DataFrames.")


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from scipy.stats.qmc import Halton  # quasi‑random sampler

# ------------------------------------------------------------------
# 1.  Feature matrix
X = shap_df.values  # N × p (8)
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)  # Gap & Silhouette both Euclidean


# ------------------------------------------------------------------
# 2.  Helper: compute Gap(k)
def gap_statistic(X, k, B=10, random_state=0):
    """
    Gap statistic for a single k.
      X : array (N × p) already scaled
      k : candidate number of clusters
      B : Monte‑Carlo reference replicates
    returns (gap, sdk)  – sdk = std dev * sqrt(1 + 1/B)   (Tibshirani 2001)
    """
    rng = np.random.RandomState(random_state)
    km = KMeans(n_clusters=k, n_init=10, random_state=rng)

    # 1. dispersion for the real data
    km.fit(X)
    Wk = km.inertia_

    # 2. dispersion for B reference datasets
    bounds = np.column_stack((X.min(axis=0), X.max(axis=0)))  # hyper‑box
    log_Wkb = np.zeros(B)
    for b in range(B):
        # Halton gives better space‑filling than pure uniform rng
        ref = Halton(d=X.shape[1], seed=rng.randint(1_000_000)).random(n=X.shape[0])
        ref = ref * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        km.fit(ref)
        log_Wkb[b] = np.log(km.inertia_)

    gap = np.mean(log_Wkb) - np.log(Wk)
    sdk = np.sqrt(1 + 1 / B) * log_Wkb.std(ddof=1)
    return gap, sdk


# ------------------------------------------------------------------
# 3.  Search over k = 2 … 15
k_range = range(2, 16)
silhouette_by_k = {}
inertia_by_k = {}
gap_by_k = {}
gap_se_by_k = {}

print("Calculating internal metrics for k in", list(k_range))
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
    gap, se = gap_statistic(X_scaled, k, B=10, random_state=42)

    inertia_by_k[k] = inertia
    silhouette_by_k[k] = sil
    gap_by_k[k] = gap
    gap_se_by_k[k] = se

    print(f"k={k:2d}  WCSS={inertia:9.1f}  Sil={sil:0.3f}  Gap={gap:0.3f} ±{se:0.3f}")


# ------------------------------------------------------------------
# 4.  Best k via Gap Statistic  (1‑SE rule)
def best_k_gap(gap, se):
    ks = sorted(gap.keys())
    for i, k in enumerate(ks[:-1]):
        if gap[k] >= gap[ks[i + 1]] - se[ks[i + 1]]:
            return k
    return ks[-1]


k_gap = best_k_gap(gap_by_k, gap_se_by_k)
k_sil = max(silhouette_by_k, key=silhouette_by_k.get)

print(f"\nGap‑statistic optimal k = {k_gap}")
print(f"Silhouette optimal k    = {k_sil}")

# ------------------------------------------------------------------
# 5.  Plot all three curves
plt.figure(figsize=(15, 4))

# Silhouette
plt.subplot(1, 3, 1)
sns.lineplot(x=list(silhouette_by_k.keys()), y=list(silhouette_by_k.values()), marker="o")
plt.axvline(k_sil, ls="--", c="grey")
plt.title("Silhouette")
plt.xlabel("k");
plt.ylabel("Avg score");
plt.grid(True)

# Elbow (WCSS)
plt.subplot(1, 3, 2)
sns.lineplot(x=list(inertia_by_k.keys()), y=list(inertia_by_k.values()), marker="o")
plt.title("Elbow (WCSS)")
plt.xlabel("k");
plt.ylabel("Inertia");
plt.grid(True)

# Gap
plt.subplot(1, 3, 3)
gaps = [gap_by_k[k] for k in k_range]
ses = [gap_se_by_k[k] for k in k_range]
plt.errorbar(k_range, gaps, yerr=ses, marker="o", capsize=3)
plt.axvline(k_gap, ls="--", c="grey")
plt.title("Gap Statistic")
plt.xlabel("k");
plt.ylabel("Gap");
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
"""
Try spectral clustering
"""
from sklearn.preprocessing   import StandardScaler                 # z-score
from sklearn.cluster         import SpectralClustering
from sklearn.pipeline        import make_pipeline
from sklearn.metrics         import silhouette_score, silhouette_samples, make_scorer
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------
# 1.  Feature matrix (all SHAP columns)
X = shap_df.values  # N × 8

# ---------------------------------------------------------------------
# 2.  Manual grid-search over k = 3, 4
best_k, best_score, best_pipe, best_labels = None, -1.0, None, None
scores = {}

for k in (4, 5):
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),  # z-score
        SpectralClustering(
            n_clusters=k,
            affinity='rbf',
            gamma=1.0,
            assign_labels='kmeans',
            random_state=42))

    labels = pipe.fit_predict(X)  # works because last step has fit_predict
    X_scaled = pipe.named_steps['standardscaler'].transform(X)
    sil = silhouette_score(X_scaled, labels)
    scores[k] = sil
    print(f'k = {k} → mean silhouette = {sil:0.3f}')

    if sil > best_score:
        best_k, best_score, best_pipe, best_labels = k, sil, pipe, labels

print(f'\nChosen k = {best_k}  (silhouette = {best_score:0.3f})')

# ---------------------------------------------------------------------
# 3.  Detailed silhouette per cluster (still on scaled space)
X_scaled = best_pipe.named_steps['standardscaler'].transform(X)
sil_indiv = silhouette_samples(X_scaled, best_labels)

print(f'\nAverage silhouette (full data): {best_score:0.3f}')
for lab in range(best_k):
    vals = sil_indiv[best_labels == lab]
    print(f'  – Cluster {lab}: N={vals.size:3d} | μ={vals.mean():0.3f} | σ={vals.std():0.3f}')

# ---------------------------------------------------------------------
# 4.  Scatter plot for one feature (raw vs SHAP), coloured by clusters
feat = 'PSG_Sleep_Dur'
plt.figure(figsize=(7, 5))
for c in range(best_k):
    idx = best_labels == c
    plt.scatter(df_SHIP.loc[idx, feat],
                shap_df.loc[idx, feat],
                s=30, label=f'Cluster {c}')
plt.xlabel(feat)
plt.ylabel('SHAP value')
plt.title(f'{feat}: raw vs. SHAP (Spectral, k={best_k})')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

feat = 'Age_at_Scan'
plt.figure(figsize=(7, 5))
for c in range(best_k):
    idx = best_labels == c
    plt.scatter(df_SHIP.loc[idx, feat],
                shap_df.loc[idx, feat],
                s=30, label=f'Cluster {c}')
plt.xlabel(feat)
plt.ylabel('SHAP value')
plt.title(f'{feat}: raw vs. SHAP (Spectral, k={best_k})')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
"""
Hierarchical (Agglomerative) clustering with Ward linkage
"""
# ----------------------------------------------------------------------
# 0. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing   import StandardScaler
from sklearn.cluster         import AgglomerativeClustering
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import silhouette_score, silhouette_samples

# ----------------------------------------------------------------------
# 1.  Feature matrix (all SHAP columns)
X_embed = shap_df.values                    # N × 8

# ----------------------------------------------------------------------
# 2.  Grid-search for the best k (3 vs 4) using silhouette on z-scored space
silhouette_by_k = {}
for k in (3, 4):
    # z-score + Ward linkage in a single pipeline
    pipe = Pipeline([
        ('scale', StandardScaler(with_mean=True, with_std=True)),
        ('agg',   AgglomerativeClustering(
                      n_clusters = k,
                      linkage    = 'ward'))
    ])
    pipe.fit(X_embed)
    labels_k = pipe.named_steps['agg'].labels_
    X_scaled = pipe.named_steps['scale'].transform(X_embed)   # same geometry as clustering
    sil      = silhouette_score(X_scaled, labels_k)
    silhouette_by_k[k] = sil
    print(f'k = {k} → avg silhouette = {sil:0.3f}')

best_k = max(silhouette_by_k, key=silhouette_by_k.get)
print(f'\nChosen k = {best_k} (highest silhouette)')

# ----------------------------------------------------------------------
# 3.  Final pipeline with the chosen k
pipeline = Pipeline([
    ('scale', StandardScaler(with_mean=True, with_std=True)),
    ('agg',   AgglomerativeClustering(
                  n_clusters = best_k,
                  linkage    = 'ward'))
])
pipeline.fit(X_embed)
labels = pipeline.named_steps['agg'].labels_
X_scaled = pipeline.named_steps['scale'].transform(X_embed)

# ----------------------------------------------------------------------
# 4.  Detailed silhouette per cluster
sil_global = silhouette_score(X_scaled, labels)
sil_indiv  = silhouette_samples(X_scaled, labels)

print(f'\nAverage silhouette: {sil_global:0.3f}')
for lab in range(best_k):
    vals = sil_indiv[labels == lab]
    print(f'  – Cluster {lab}: N={vals.size:3d} | μ={vals.mean():0.3f} | σ={vals.std():0.3f}')

# ----------------------------------------------------------------------
# 5.  Scatter plot: raw vs SHAP for two illustrative features
for feat in ['PSG_Sleep_Dur', 'Age_at_Scan']:
    plt.figure(figsize=(7, 5))
    x = df_SHIP[feat].values
    y = shap_df[feat].values
    for c in range(best_k):
        idx = labels == c
        plt.scatter(x[idx], y[idx], s=30, label=f'Cluster {c}')
    plt.xlabel(feat)
    plt.ylabel('SHAP value')
    plt.title(f'{feat}: raw vs. SHAP (Hierarchical, k={best_k})')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# %%
# set plot size
ax = shapiq.plot.bar_plot(
    ivs,
    feature_names=X,
    abbreviate=False,
    show=False            # returns the Axes
)

ax.figure.set_size_inches(8, 10)   # or (6.8, 3.5) for double-column width
ax.figure.tight_layout()
plt.show()

# %%
"""
backup of plot_SHAP_cluster_SHAP-IQ_AutoGluon_SHIP.py
"""
# %%
"""
1. Use GMM compute BIC to select the best k
2. Based on best k compare the results for KMeans, Spectral Clustering, and GMM
3. Compare the results of KMeans, Spectral Clustering, and GMM by Gap statistic and silhouette score plots
"""


def _reference_inertia_once(X, k, random_state):
    """One bootstrap replicate on a uniform reference distribution."""
    rng = np.random.RandomState(random_state)
    random_data = rng.uniform(low=X.min(axis=0),
                              high=X.max(axis=0),
                              size=X.shape)
    km = KMeans(
        n_clusters=k,
        n_init=50,
        max_iter=500,
        random_state=42
    )
    km.fit(random_data)
    return km.inertia_


def compute_gap_statistic(X, k_range, n_replicates=10):
    """
    Returns:
        gaps          : dict {k: gap}
        sdkhat        : dict {k: s_k * sqrt(1 + 1/n_replicates)}
        reference_log : dict {k: E[log(W*_k)]} – optional for inspection
    """
    X = check_array(X)  # defensive copy, ensures contiguous
    gaps, sdkhat, reference_log = {}, {}, {}

    for k in k_range:
        # Inertia on observed data
        km = KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=500,
            random_state=42
        )
        km.fit(X)
        inertia_orig = km.inertia_

        print("\nstarting parallel bootstrap for k =", k)
        # Parallel bootstrap on reference data
        register_htcondor("INFO")
        with parallel_config(
                backend="htcondor",
                pool="head2.htc.inm7.de",
                request_cpus=1,
                request_disk="1GB",
                request_memory="2Gb",
                throttle=n_replicates,  # Throttle for the outer jobs only
                export_metadata=True,
                shared_data_dir=joblib_htcondor_path,
                log_dir_prefix=joblib_log_path,
                n_jobs=-1,
        ):
            ref_inertias = Parallel()(
                delayed(_reference_inertia_once)(X, k, 42 + i)
                for i in range(n_replicates)
            )

        log_wk_ref = np.log(ref_inertias)  # vector of log(W*_kb)
        ref_mean = log_wk_ref.mean()  # mean_log, ȷ̄   (bar-l)

        # gap = ref_mean - np.log(inertia_orig)
        # sk = ref_std * np.sqrt(1 + 1. / n_replicates)
        gap = ref_mean - np.log(inertia_orig)
        sd_k_value = np.sqrt(((log_wk_ref - ref_mean) ** 2).mean())

        gaps[k], sdkhat[k] = gap, sd_k_value

    return gaps, sdkhat

# %%
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
    # *** FIX: Correctly initialize Pipeline with steps ***
    pipe = Pipeline([
        # ('scale', StandardScaler()),
        ('kmeans', KMeans(
            n_clusters=k,
            n_init=50, # Lower n_init for faster grid search
            max_iter=500,
            random_state=42,  # 42
        ))
    ])
    # labels_k = pipe.fit_predict(X_embed)
    kmeans_model = pipe.fit(X_embed)  # Fit the pipeline
    labels_k = kmeans_model.predict(X_embed)  # Predict labels
    inertia = kmeans_model.named_steps['kmeans'].inertia_
    inertia_by_k[k] = inertia
    print(f'k = {k} → Inertia = {inertia:.2f}', end='')  # Print inertia first

    # Ensure there's more than 1 cluster label before calculating silhouette score
    if len(set(labels_k)) > 1:
        sil = silhouette_score(X_embed, labels_k)

        # X_scaled_for_eval = kmeans_model.named_steps['scale'].transform(X_embed)
        # # Calculate silhouette score using the scaled data and the labels
        # sil = silhouette_score(X_scaled_for_eval, labels_k)
        silhouette_by_k[k] = sil
        print(f'k = {k} → avg silhouette = {sil:0.3f}')

        if sil > 0:  # plot only if ≥2 clusters
            sil_vals = silhouette_samples(X_embed, labels_k)

            fig, ax1 = plt.subplots(figsize=(6, 4))
            y_lower = 10
            for i in range(k):
                ith_vals = sil_vals[labels_k == i]
                ith_vals.sort()
                size_i = ith_vals.shape[0]
                y_upper = y_lower + size_i

                color = cm.nipy_spectral(float(i) / k)  # nice distinct colours
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0, ith_vals,
                    facecolor=color, edgecolor=color, alpha=0.7
                )
                # Label the cluster number in the middle
                ax1.text(-0.05, y_lower + 0.5 * size_i, str(i))
                y_lower = y_upper + 10  # 10‑pt gap between clusters

            ax1.set_title(f"Silhouette plot — k = {k}")
            ax1.set_xlabel("Silhouette coefficient")
            ax1.set_ylabel("Cluster")

            # Red dashed line for the global average silhouette score
            ax1.axvline(x=sil, color="red", linestyle="--",
                        label=f"avg = {sil:0.3f}")
            ax1.set_yticks([])  # bar plot already encodes y
            ax1.set_xlim([-0.1, 1])
            ax1.legend(loc="lower right", frameon=False)
            plt.tight_layout()

            sil_path = SHAP_clustering_path + f"silhouette_k{k}.png"
            plt.savefig(sil_path, dpi=300, bbox_inches="tight")
            # plt.show()
            plt.close(fig)  # keep memory footprint low
            print(f"  → silhouette figure saved to {sil_path}")
    else:
        print(f'k = {k} → Only 1 cluster found, cannot calculate silhouette score.')
        silhouette_by_k[k] = -1 # Assign a low score or handle as needed

    # BIC calculation (Gaussian Mixture Model)
    gmm = GaussianMixture(n_components=k,
                          covariance_type='full',  # spherical
                          n_init=10,
                          random_state=42,
                          )
    bic = gmm.fit(X_embed).bic(X_embed)
    bic_by_k[k] = bic
    print(f' | BIC = {bic:0.1f}')

# %%
# ----------------------------------------------------------------------
# 2c. Gap‑statistic
gap_values, gap_se = compute_gap_statistic(X_embed, k_range, n_replicates=20)
print("\nGap statistic:")
for k in k_range:
    print(f"k={k:2d}  gap={gap_values[k]:6.3f}  s_k={gap_se[k]:6.3f}")
# Save the gap values
gap_values_path = SHAP_clustering_path + 'gap_values.pkl'
with open(gap_values_path, 'wb') as f:
    pickle.dump(gap_values, f)
gap_se_path = SHAP_clustering_path + 'gap_se.pkl'
with open(gap_se_path, 'wb') as f:
    pickle.dump(gap_se, f)

# ----------------------------------------------------------------------
# 2b. Plot Silhouette scores vs. k using Seaborn
# plt.figure(figsize=(8, 5))
# *** Use seaborn.lineplot ***
# sns.lineplot(x=list(silhouette_by_k.keys()), y=list(silhouette_by_k.values()), marker='o')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Average Silhouette Score")
# plt.title("Silhouette Score vs. Number of Clusters (k)")
# plt.xticks(list(k_range)) # Ensure all k values are shown as ticks
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(5, 7)) # Make figure wider for two subplots
plt.figure(figsize=(5, 13))   # a bit taller
# Plot Silhouette Score
plt.subplot(4, 1, 1)
sns.lineplot(x=list(silhouette_by_k.keys()), y=list(silhouette_by_k.values()), marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.title("Silhouette Score vs. k")
plt.xticks(list(k_range)) # Ensure all k values are shown as ticks
plt.grid(True)

# Plot Elbow Method (Inertia)
plt.subplot(4, 1, 2)
sns.lineplot(x=list(inertia_by_k.keys()), y=list(inertia_by_k.values()), marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method (Inertia vs. k)")
plt.xticks(list(k_range)) # Ensure all k values are shown as ticks
plt.grid(True)

plt.subplot(4, 1, 3)
sns.lineplot(x=list(gap_values.keys()),
             y=list(gap_values.values()), marker='o')
plt.title("Gap Statistic")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Gap")
plt.xticks(list(k_range))
plt.grid(True)

plt.subplot(4, 1, 4)
sns.lineplot(x=list(bic_by_k.keys()),
             y=list(bic_by_k.values()), marker='o')
plt.title("BIC (Gaussian Mixture)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("BIC  (lower is better)")
plt.xticks(list(k_range))
plt.grid(True)

plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
# save the figure
plt.savefig(SHAP_clustering_path + 'SHAP_clustering_statistic.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# ----------------------------------------------------------------------
# 3. Select best k and run final pipeline
# if not silhouette_by_k or all(score == -1 for score in silhouette_by_k.values()):
#     print("\nError: No valid silhouette scores calculated. Cannot determine best k.")
#     # Handle error case, e.g., exit or set a default k
#     best_k = 3 # Defaulting to 3 as an example
#     print(f"Warning: Proceeding with default k = {best_k}")
# else:
#     # Find the k that yielded the highest silhouette score
#     best_k = max(silhouette_by_k, key=silhouette_by_k.get)
#     print(f'\nChosen k = {best_k} (highest silhouette score: {silhouette_by_k[best_k]:.3f})')
def _argmin(d):
    """Return key with smallest value; None if dict empty or all NaNs."""
    if not d:
        return None
    # Filter out nan / inf in case GMM failed for a k
    valid = {k: v for k, v in d.items() if np.isfinite(v)}
    return min(valid, key=valid.get) if valid else None

# 1️⃣  Try BIC first
best_k = _argmin(bic_by_k)

# 2️⃣  Fallback to silhouette if BIC missing
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

# Final pipeline with the chosen k
pipeline = Pipeline([
    # ('scale',  StandardScaler()),
    ('kmeans', KMeans(
        n_clusters=best_k,  # best_k
        init='k-means++',
        n_init=50, # Use higher n_init for the final model
        max_iter=500,
        random_state=42,  # 42
        verbose=0
    ))
])
labels = pipeline.fit_predict(X_embed)

# ----------------------------------------------------------------------
# 4. Detailed silhouette per cluster for the chosen k
# Check if clustering resulted in more than one label
if len(np.unique(labels)) > 1:
    sil_global = silhouette_score(X_embed, labels)
    sil_indiv = silhouette_samples(X_embed, labels)

    print(f'\nAverage silhouette for k={best_k}: {sil_global:0.3f}')
    # Use np.unique to handle cases where cluster labels might not be contiguous
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        # Filter out potential noise points if any algorithm other than K-Means were used
        if lab == -1: continue
        vals = sil_indiv[labels == lab]
        # Check if cluster is not empty before calculating mean/std
        if vals.size > 0:
             print(f'  – Cluster {lab}: N={vals.size:3d} | μ={vals.mean():0.3f} | σ={vals.std():0.3f}')
        else:
             print(f'  – Cluster {lab}: N=0')
else:
    print(f'\nClustering with k={best_k} resulted in only one cluster or failed. Cannot calculate detailed silhouette scores.')

# -----------------------------------------------------------------------
# 4.5 best k suggest by gap statistic
def gap_optimal_k(gap_dict, sk_dict):
    # Ensure k are sorted
    ks = sorted(gap_dict.keys())
    for i, k in enumerate(ks[:-1]):          # up to penultimate
        if gap_dict[k] >= gap_dict[ks[i+1]] - sk_dict[ks[i+1]]:
            return k
    return ks[-1]                            # fallback to max k

gap_best_k = gap_optimal_k(gap_values, gap_se)
print(f"\nGap‑statistic suggests k = {gap_best_k}")

# ----------------------------------------------------------------------
# 5. Scatter plots for specific features (using the final labels from best_k)
#    (Using seaborn for scatter plots as well for consistency)

# Plot for 'PSG_Sleep_Dur'
feat1 = 'PSG_Sleep_Dur'
if feat1 in df_SHIP.columns and feat1 in shap_df.columns:
    plot_df1 = pd.DataFrame({
        'Original Value': df_SHIP[feat1].values,
        'SHAP Value': shap_df[feat1].values,
        'Cluster': labels.astype(str) # Convert labels to string for categorical plotting
    })

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df1, x='Original Value', y='SHAP Value', hue='Cluster', palette='viridis', s=50)
    plt.xlabel(feat1 + " (Original Value)")
    plt.ylabel('SHAP value for ' + feat1)
    plt.title(f'{feat1}: Original vs. SHAP (K-means, k={best_k})')
    plt.legend(title='Cluster', frameon=False)
    plt.tight_layout()
    # Save the figure
    plt.savefig(SHAP_clustering_path + f'SHAP_clustering_{feat1}.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print(f"Feature '{feat1}' not found in both DataFrames.")

# Plot for 'Age_at_Scan'
feat2 = 'Age_at_Scan'
if feat2 in df_SHIP.columns and feat2 in shap_df.columns:
    plot_df2 = pd.DataFrame({
        'Original Value': df_SHIP[feat2].values,
        'SHAP Value': shap_df[feat2].values,
        'Cluster': labels.astype(str) # Convert labels to string
    })

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df2, x='Original Value', y='SHAP Value', hue='Cluster', palette='viridis', s=50)
    plt.xlabel(feat2 + " (Original Value)")
    plt.ylabel('SHAP value for ' + feat2)
    plt.title(f'{feat2}: Original vs. SHAP (K-means, k={best_k})')
    plt.legend(title='Cluster', frameon=False)
    plt.tight_layout()
    # Save the figure
    plt.savefig(SHAP_clustering_path + f'SHAP_clustering_{feat2}.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print(f"Feature '{feat2}' not found in both DataFrames.")

# %%
import umap.umap_ as umap
print("\nComputing a 2‑D UMAP projection…")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=50,        # smaller ⇒ focus on local; larger ⇒ preserve global
    # min_dist=0.05,         # 0 ≈ tighter clusters, 0.5 ≈ looser
    metric="euclidean",    # change to "cosine" if you prefer
    random_state=42
)
embedding = reducer.fit_transform(X_embed)   # shape (n_samples, 2)

# Make a DataFrame just for plotting
umap_df = pd.DataFrame({
    "UMAP‑1": embedding[:, 0],
    "UMAP‑2": embedding[:, 1],
    "Cluster": labels.astype(str)   # strings → categorical colouring
})

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=umap_df,
    x="UMAP‑1", y="UMAP‑2",
    hue="Cluster",
    palette="viridis",
    s=50, alpha=0.9,
    linewidth=0
)
plt.title(f"UMAP projection of SHAP space  (k‑means, k = {best_k})")
plt.axis("off")             # looks cleaner for embeddings; comment out if you want axes
plt.legend(title="Cluster", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

umap_path = SHAP_clustering_path + "SHAP_clustering_UMAP.png"
plt.savefig(umap_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"UMAP figure saved to: {umap_path}")

# %%
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.mixture  import GaussianMixture
from sklearn.metrics  import (silhouette_score,
                              calinski_harabasz_score,
                              davies_bouldin_score)

best_k = 4

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
        print(f"  ⚠️  {name} produced only one cluster — metrics undefined.")
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
    print(f"\n⭐  Best algorithm by Silhouette → {winner}")
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

    # 1️⃣  Feature‑1 scatter
    for feat in ["PSG_Sleep_Dur", "Age_at_Scan"]:
        if feat in df_SHIP.columns and feat in shap_df.columns:
            df_plot = pd.DataFrame({
                "Original": df_SHIP[feat].values,
                "SHAP":    shap_df[feat].values,
                "Cluster": lbls.astype(str)
            })
            plt.figure(figsize=(7, 5))
            sns.scatterplot(
                data=df_plot, x="Original", y="SHAP",
                hue="Cluster", palette="viridis", s=50, linewidth=0
            )
            plt.xlabel(f"{feat} (Original)")
            plt.ylabel(f"SHAP value for {feat}")
            plt.title(f"{feat}: Original vs. SHAP  ({algo_name}, k = {best_k})")
            plt.legend(title="Cluster", frameon=False,
                       bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            fname = f"{SHAP_clustering_path}{algo_name}_{feat}.png"
            # plt.savefig(fname, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
            # print(f"  → saved {Path(fname).name}")

    # 2️⃣  2‑D UMAP embedding
    reducer = umap.UMAP(
        n_components=2, n_neighbors=50,
        metric="euclidean", random_state=42
    )
    emb = reducer.fit_transform(X_embed)
    df_umap = pd.DataFrame({
        "UMAP‑1": emb[:, 0], "UMAP‑2": emb[:, 1],
        "Cluster": lbls.astype(str)
    })
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df_umap, x="UMAP‑1", y="UMAP‑2",
        hue="Cluster", palette="viridis",
        s=50, alpha=0.9, linewidth=0
    )
    plt.title(f"UMAP projection  ({algo_name}, k = {best_k})")
    plt.axis("off")
    plt.legend(title="Cluster", frameon=False,
               bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fname = f"{SHAP_clustering_path}{algo_name}_UMAP.png"
    # plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    # print(f"  → saved {Path(fname).name}")

# ── run the helper for every algorithm ─────────────────────────────────
for algo_name, lbls in labels_dict.items():
    visualise_algo(algo_name, lbls)
