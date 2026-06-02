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
import pickle
from joblib import load
from datetime import datetime

import matplotlib as mpl
# Tell Matplotlib: “Whenever I ask for sans-serif, try Arial first”
mpl.rcParams['font.family']      = 'sans-serif'
mpl.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='SHAP for AutoGluon, SHIP_Trend dataset.')
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

print(f"\nStarting SHAP for AutoGluon pipeline for {target} prediction with feature combination {feature_comb}.\n")

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

# load saved shap value by pickle
print(f"\nLoading SHAP values for {feature_comb}, {target}.\n")
shap_values_saved_path = SHAP_path + 'explanation_train_set_train.pkl'
with open(shap_values_saved_path, 'rb') as f:
    explanation_train_set_train = pickle.load(f)

explanation = explanation_train_set_train

# %%
"""
Feature < 20 case
"""
# aim is only to get feature orders
plt.clf()
# Create a new figure with custom size
# fig, ax = plt.subplots(figsize=(10, 20))  # Adjust width (15) and height (10) as needed
# shap.plots.bar(explanation, max_display=100, clustering_cutoff=0.6, show=False, ax=ax)
# Generate the SHAP bar plot and pass the custom axis
# shap.plots.bar(explanation, max_display=5, clustering_cutoff=0.5, show=False)
shap.plots.bar(explanation, clustering_cutoff=0.58, show=False)

plt.tight_layout()

# Extract yticklabels
ax = plt.gca()
yticklabels = [label.get_text() for label in ax.get_yticklabels()]
unique_yticklabels = yticklabels[:len(yticklabels) // 2]  # Remove duplicates
print("Unique yticklabels:", unique_yticklabels)

plt.close()

feature_order = [explanation.feature_names.index(label) for label in unique_yticklabels]

# %%
# scatter plot
importance_df = pd.DataFrame({
    'feature': X,
    'importance': np.abs(explanation.values).mean(axis=0)
})
# select top 10 features to a list
top_10_features = importance_df.sort_values(by='importance', ascending=False).head(10)['feature'].tolist()
for feature in top_10_features:
    # for color in ['Age_at_Scan', 'PSG_Sleep_Dur', 'Self_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Eff']:
    for color in ['Age_at_Scan']:
        # if there's no such feature in X, skip
        if color not in X:
            continue
        shap.plots.scatter(explanation[:, feature], color=explanation[:, color], show=False)
        plt.title(feature)
        plt.tight_layout()
        # plt.savefig(save_path + f"{feature}_{color}_scatter.png")
        plt.show()
        plt.close()

# %%
import numpy as np
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm
from sklearn.metrics import r2_score


def shap_dependence_curve(
    explanation,
    feature,
    color=None,
    degree=1,
    ci=0.95,
    n_grid=200,
    point_alpha=0.7,
    point_size=16,
    fit_color="crimson",
    ci_color="crimson",
    ci_alpha=0.18,
    annotate=True,
    show=True,
    ax=None,
):
    """
    Draw a SHAP dependence plot (scatter) with a fitted polynomial curve,
    confidence interval, R^2, and p-value.

    Parameters
    ----------
    explanation : shap.Explanation
        SHAP Explanation object.
    feature : str or int
        Feature to show on x-axis (and whose SHAP values are shown on y-axis).
    color : str or int or None, default=None
        Feature used for coloring the scatter points, e.g. explanation[:, color].
        If None, no coloring feature is used.
    degree : int, default=1
        Degree of polynomial fit. degree=1 gives linear regression,
        degree=2 gives quadratic, etc.
    ci : float, default=0.95
        Confidence interval level for the fitted mean.
    n_grid : int, default=200
        Number of points used to draw the fitted curve.
    point_alpha : float, default=0.7
        Scatter point transparency.
    point_size : float, default=16
        Scatter point size.
    fit_color : str, default="crimson"
        Color of fitted curve.
    ci_color : str, default="crimson"
        Color of CI band.
    ci_alpha : float, default=0.18
        Transparency of CI band.
    annotate : bool, default=True
        Whether to annotate R^2 and p-value on the plot.
    show : bool, default=True
        Whether to call plt.show().
    ax : matplotlib.axes.Axes or None, default=None
        Existing axis to draw on.

    Returns
    -------
    fig, ax, model
        Figure, axis, and fitted statsmodels OLS result object.
    """
    # --- extract x (feature values) and y (SHAP values) ---
    x = np.asarray(explanation[:, feature].data).reshape(-1)
    y = np.asarray(explanation[:, feature].values).reshape(-1)

    # keep finite values only
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < degree + 2:
        raise ValueError(
            f"Not enough valid points ({x.size}) for polynomial degree={degree}."
        )

    # --- prepare axis ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # --- draw SHAP scatter ---
    # Use a temporary shap.Explanation with the same mask so scatter matches fitted data
    exp_feat = explanation[:, feature]
    exp_feat = shap.Explanation(
        values=np.asarray(exp_feat.values).reshape(-1)[mask],
        base_values=None if exp_feat.base_values is None else np.asarray(exp_feat.base_values).reshape(-1)[mask],
        data=np.asarray(exp_feat.data).reshape(-1)[mask],
        feature_names=exp_feat.feature_names,
    )

    if color is not None:
        exp_color = explanation[:, color]
        exp_color = shap.Explanation(
            values=np.asarray(exp_color.values).reshape(-1)[mask],
            base_values=None if exp_color.base_values is None else np.asarray(exp_color.base_values).reshape(-1)[mask],
            data=np.asarray(exp_color.data).reshape(-1)[mask],
            feature_names=exp_color.feature_names,
        )
        shap.plots.scatter(
            exp_feat,
            color=exp_color,
            show=False,
            ax=ax,
            alpha=point_alpha,
            dot_size=point_size,
        )
    else:
        shap.plots.scatter(
            exp_feat,
            show=False,
            ax=ax,
            alpha=point_alpha,
            dot_size=point_size,
        )

    # --- polynomial regression using statsmodels OLS ---
    # design matrix: [1, x, x^2, ..., x^degree]
    X = np.column_stack([x**i for i in range(1, degree + 1)])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    # prediction grid
    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), n_grid)
    X_grid = np.column_stack([x_grid**i for i in range(1, degree + 1)])
    X_grid = sm.add_constant(X_grid)

    pred = model.get_prediction(X_grid).summary_frame(alpha=1 - ci)

    y_fit = pred["mean"].to_numpy()
    y_low = pred["mean_ci_lower"].to_numpy()
    y_high = pred["mean_ci_upper"].to_numpy()

    # --- overlay fitted curve and CI ---
    ax.plot(x_grid, y_fit, color=fit_color, lw=2, label=f"Poly fit (degree={degree})")
    ax.fill_between(x_grid, y_low, y_high, color=ci_color, alpha=ci_alpha, label=f"{int(ci * 100)}% CI")

    # --- annotate R^2 and p-value ---
    if annotate:
        r2 = model.rsquared
        pval = model.f_pvalue  # overall model p-value (F-test)

        if pval < 1e-3:
            p_text = f"{pval:.2e}"
        else:
            p_text = f"{pval:.3f}"

        ax.text(
            0.02,
            0.98,
            rf"$R^2={r2:.3f}$" + "\n" + rf"$p={p_text}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    ax.set_title(str(feature))
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, model

# %%
# with interaction coloring
shap_dependence_curve(
    explanation=explanation,
    feature=feature,
    color=color,
    degree=2,   # try 1 for linear, 2 for quadratic
    ci=0.95
)

# without color feature
shap_dependence_curve(
    explanation=explanation,
    feature=feature,
    color=None,
    degree=1
)

# %%
for feature in top_10_features:
    # for color in ['Age_at_Scan', 'PSG_Sleep_Dur', 'Self_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Eff']:
    for color in ['Age_at_Scan']:
        # if there's no such feature in X, skip
        if color not in X:
            continue
        shap_dependence_curve(
            explanation=explanation,
            feature=feature,
            color=color,
            degree=2,  # try 1 for linear, 2 for quadratic
            ci=0.95
        )

# %%
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
import umap

reducer = umap.UMAP(
    n_neighbors=15,      # Controls local vs global focus
    min_dist=0.1,        # Controls tightness of embedding
    n_components=2,      # Target dimension (e.g., 2 for visualization)
    metric='euclidean',  # Distance metric in SHAP space
    random_state=42      # For reproducibility
)
embedding = reducer.fit_transform(shap_df)

# %%
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10, # Minimum size for a group to be considered a cluster
    min_samples=None,    # Controls conservativeness (None often works well)
    metric='euclidean',
    cluster_selection_epsilon=0.0 # Optional: distance threshold for merging clusters
)
cluster_labels = clusterer.fit_predict(embedding) # Use shap_matrix if clustering directly

# %%
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=cluster_labels,
    s=5, # Adjust point size
    cmap='Spectral' # Choose a suitable colormap
)
plt.title('UMAP Projection of SHAP Explanations, Colored by HDBSCAN Clusters')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
# Add legend if needed
unique_labels = set(cluster_labels)
if -1 in unique_labels: # Handle noise points if present
    unique_labels.remove(-1)
    plt.scatter(embedding[cluster_labels == -1, 0], embedding[cluster_labels == -1, 1],
                c='gray', s=1, label='Noise') # Plot noise points separately
# Create legend for actual clusters
#... (code to create legend based on scatter handles/labels)...
plt.legend()
plt.show()

# %%
"""
1
"""
# 创建主图（用来画蜂巢图）
# fig, ax1 = plt.subplots(figsize=(8, 10))
# 在主图上绘制蜂巢图，并保留热度条
shap.summary_plot(explanation, df_SHIP[X], feature_names=X, plot_type="dot", show=False, plot_size=(8, 10), sort=True)
# plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，留出右侧空间放热度条
# 获取共享的 y 轴
ax1 = plt.gca()
# 创建共享 y 轴的另一个图，绘制特征贡献图在顶部x轴
ax2 = ax1.twiny()
shap.summary_plot(explanation, df_SHIP[X], feature_names=X, plot_type="bar", show=False, plot_size=(8, 10), sort=True)
# plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，与蜂巢图对齐
# 在顶部 X 轴添加一条横线
ax2.axhline(y=8, color='gray', linestyle='-', linewidth=1)  # 注意y值应该对应顶部
# 调整透明度
bars = ax2.patches  # 获取所有的柱状图对象
for bar in bars:
    bar.set_alpha(0.2)  # 设置透明度
# 设置两个x轴的标签
ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12)
ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)
# 移动顶部的 X 轴，避免与底部 X 轴重叠
ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
ax2.xaxis.tick_top()  # 将刻度也移动到顶部
# 设置y轴标签
ax1.set_ylabel('Features', fontsize=12)

# Extract existing yticklabels
yticklabels = [label.get_text() for label in ax1.get_yticklabels()]
# Define label replacements
label_replacements = {
    "Age_at_Scan": "Age",
    "Depression_score": "Depression"
}
# Apply replacements
updated_yticklabels = [label_replacements.get(label, label) for label in yticklabels]
# Set new y-axis labels
ax1.set_yticklabels(updated_yticklabels)


plt.tight_layout()
plt.savefig(SHAP_path + f"AutoGluon_{target}_{feature_comb}_SHAP_combined.svg", format='svg')
plt.show()
plt.close()

# %%
"""
2
"""
shap.plots.bar(explanation, max_display=20, clustering_cutoff=0.58, show=False)
plt.tight_layout()
plt.show()
plt.close()

# %%
shap.plots.beeswarm(explanation, max_display=20, cluster_threshold=0.58, show=False)
plt.tight_layout()
plt.show()
plt.close()

# %%
shap.summary_plot(explanation, df_SHIP[X], feature_names=X, plot_type="dot", show=False, plot_size=(8, 10), sort=True)
plt.tight_layout()
plt.show()
plt.close()

# %%
"""
3
"""
fig, ax1 = plt.subplots(figsize=(8, 10))
shap.plots.bar(explanation, clustering_cutoff=0.8, show=False, ax=ax1, order=feature_order)
# ax1.axhline(y=8, color='gray', linestyle='-', linewidth=1)  # 注意y值应该对应顶部
ax1.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)
# 移动顶部的 X 轴，避免与底部 X 轴重叠
ax1.xaxis.set_label_position('top')  # 将标签移动到顶部
ax1.xaxis.tick_top()  # 将刻度也移动到顶部

# plt.tight_layout()
plt.show()
plt.close()

# %%
"""
4
"""
# Generate the beeswarm plot on the pre-sized figure
shap.plots.beeswarm(explanation, clustering=False, show=False, plot_size=(8, 10), order=feature_order)

# Get current axis
ax1 = plt.gca()

# Extract existing yticklabels
yticklabels = [label.get_text() for label in ax1.get_yticklabels()]
# Define label replacements
label_replacements = {
    "Age_at_Scan": "Age",
    "Depression_score": "Depression"
}
# Apply replacements
updated_yticklabels = [label_replacements.get(label, label) for label in yticklabels]
# Set new y-axis labels
ax1.set_yticklabels(updated_yticklabels)

# 创建共享 y 轴的另一个图，绘制特征贡献图在顶部x轴
ax2 = ax1.twiny()
# fig, ax2 = plt.subplots(figsize=(8, 10))
shap.plots.bar(explanation, clustering_cutoff=0.8, show=False, ax=ax2, order=feature_order)
# plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，与蜂巢图对齐
# 在顶部 X 轴添加一条横线
ax2.axhline(y=8, color='gray', linestyle='-', linewidth=1)  # 注意y值应该对应顶部
# 调整透明度
bars = ax2.patches  # 获取所有的柱状图对象
for bar in bars:
    bar.set_alpha(0.2)  # 设置透明度
# 设置两个x轴的标签
ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12)
ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)
# 移动顶部的 X 轴，避免与底部 X 轴重叠
ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
ax2.xaxis.tick_top()  # 将刻度也移动到顶部
# 设置y轴标签
ax1.set_ylabel('Features', fontsize=12)
plt.tight_layout()
# plt.savefig("SHAP_combined_with_top_line_corrected.pdf", format='pdf', bbox_inches='tight')
plt.show()
plt.close()


# %%
# bar plot for Owen value first
plt.clf()
fig, ax1 = plt.subplots(figsize=(8, 10))
# Create a new figure with custom size
# fig, ax = plt.subplots(figsize=(10, 20))  # Adjust width (15) and height (10) as needed
# shap.plots.bar(explanation, max_display=100, clustering_cutoff=0.6, show=False, ax=ax)
# Generate the SHAP bar plot and pass the custom axis
# shap.plots.bar(explanation, max_display=5, clustering_cutoff=0.5, show=False)
shap.plots.bar(explanation, clustering_cutoff=0.8, show=False, ax=ax1)


plt.tight_layout()

# Extract yticklabels
ax = plt.gca()
yticklabels = [label.get_text() for label in ax.get_yticklabels()]
unique_yticklabels = yticklabels[:len(yticklabels) // 2]  # Remove duplicates
print("Unique yticklabels:", unique_yticklabels)

# Show the enlarged plot
plt.show()
plt.close()

# %%
# Define the number of features to display
top_n = 5  # Show the top 10 features

# Select the indices of the top N features by mean absolute SHAP value
top_features = np.argsort(np.abs(explanation.values).mean(axis=0))[-top_n:][::-1]

# Create a new explanation object with only the top N features
top_explanation = shap.Explanation(
    values=explanation.values[:, top_features],
    base_values=explanation.base_values,
    data=explanation.data[:, top_features],
    feature_names=np.array(explanation.feature_names)[top_features],
)

# Create the bar plot for the top features without "sum of others"
shap.plots.bar(top_explanation, clustering=False, show=False)

plt.show()
plt.close()

# %%
# Map unique_yticklabels back to their indices
# feature_order = [explanation.feature_names.index(label) for label in unique_yticklabels]

# Create the beeswarm plot with the same feature order
# shap.plots.beeswarm(explanation, order=feature_order, clustering=False, show=False, color='RdYlBu_r')

# %%
# Generate the beeswarm plot on the pre-sized figure
shap.plots.beeswarm(explanation, clustering=False, show=False, plot_size=(8, 10))

# Get current axis
ax = plt.gca()

# Extract existing yticklabels
yticklabels = [label.get_text() for label in ax.get_yticklabels()]

# Define label replacements
label_replacements = {
    "Age_at_Scan": "Age",
    "Depression_score": "Depression"
}

# Apply replacements
updated_yticklabels = [label_replacements.get(label, label) for label in yticklabels]

# Set new y-axis labels
ax.set_yticklabels(updated_yticklabels)

plt.tight_layout()
# if title is not None:
#     plt.title(title)

# Save and close
# plt.savefig(SHAP_path + f"AutoGluon_{target}_{feature_comb}_beeswarm.svg", format='svg', dpi=1200, bbox_inches='tight')
plt.show()
plt.close()

# %%
# scatter plot
importance_df = pd.DataFrame({
    'feature': X,
    'importance': np.abs(explanation.values).mean(axis=0)
})
# select top 10 features to a list
top_10_features = importance_df.sort_values(by='importance', ascending=False).head(10)['feature'].tolist()
for feature in top_10_features:
    # for color in ['Age_at_Scan', 'PSG_Sleep_Dur', 'Self_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Eff']:
    for color in ['Age_at_Scan']:
        # if there's no such feature in X, skip
        if color not in X:
            continue
        shap.plots.scatter(explanation[:, feature], color=explanation[:, color], show=False)
        plt.title(feature)
        plt.tight_layout()
        # save as svg
        plt.savefig(SHAP_path + f"scatter_{feature}_color_{color}.svg", format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
        plt.close()