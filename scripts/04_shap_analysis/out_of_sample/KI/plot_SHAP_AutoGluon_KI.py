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
parser = argparse.ArgumentParser(description='SHAP for AutoGluon, KI dataset.')
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
# targets = feature_lists[9]

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
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/{target}/'

case_results_path = results_path + f"{feature_comb}/"
SHAP_path = case_results_path + 'SHAP/'

# load saved shap value by pickle
print(f"\nLoading SHAP values for {feature_comb}, {target}.\n")
shap_values_saved_path = SHAP_path + 'explanation_train_set_test_KI.pkl'
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
shap.plots.bar(explanation, clustering_cutoff=0.8, show=False)

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
"""
1
"""
# 创建主图（用来画蜂巢图）
# fig, ax1 = plt.subplots(figsize=(8, 10))
# 在主图上绘制蜂巢图，并保留热度条
shap.summary_plot(explanation, feature_names=X, plot_type="dot", show=False, plot_size=(8, 10), sort=True)
# plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，留出右侧空间放热度条
# 获取共享的 y 轴
ax1 = plt.gca()
# 创建共享 y 轴的另一个图，绘制特征贡献图在顶部x轴
ax2 = ax1.twiny()
shap.summary_plot(explanation, feature_names=X, plot_type="bar", show=False, plot_size=(8, 10), sort=True)
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
# plt.savefig(SHAP_path + f"AutoGluon_{target}_{feature_comb}_SHAP_combined.svg", format='svg')
plt.show()
plt.close()
