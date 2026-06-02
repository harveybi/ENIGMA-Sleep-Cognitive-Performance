import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %%
'''
Ridge model results for Liege dataset
'''
ridge_results_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/Ridge/'
target_list = ['1-back', '2-back', '3-back', 'Memory', 'Stroop']
feature_list = ["Sleep", "Cov", "Sleep_Cov"]

df_ridge_Liege_results = None
for feature in feature_list:
    for target in target_list:
        csv_file = ridge_results_path + target + '/Liege/' + feature + '/results.csv'
        df = pd.read_csv(csv_file, index_col=0)
        if df_ridge_Liege_results is None:
            df_ridge_Liege_results = df
        else:
            df_ridge_Liege_results = pd.concat([df_ridge_Liege_results, df], axis=0)

df_ridge_Liege_results.reset_index(inplace=True, drop=True)

# %%
column_names = df_ridge_Liege_results.columns.tolist()
average_cv_test_r2_column = [col for col in column_names if "Average CV Test R2" in col]

heatmap_data = df_ridge_Liege_results.pivot("Feature Combination", "Target", average_cv_test_r2_column[0])

# Generate the heatmap
# plt.figure()
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='RdBu_r', vmin=-0.4, vmax=0.4)
plt.title('Heatmap of Average CV Test R2, Ridge, Liege dataset')
plt.tight_layout()
plt.show()
plt.close()

# %%
"""
SVM-rbf model results for Liege dataset
"""
ridge_results_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/SVM-rbf/'
target_list = ['1-back', '2-back', '3-back', 'Memory', 'Stroop']
feature_list = ["Sleep", "Cov", "Sleep_Cov"]

df_SVM_rbf_Liege_results = None
for feature in feature_list:
    for target in target_list:
        csv_file = ridge_results_path + target + '/Liege/' + feature + '/results.csv'
        df = pd.read_csv(csv_file, index_col=0)
        if df_SVM_rbf_Liege_results is None:
            df_SVM_rbf_Liege_results = df
        else:
            df_SVM_rbf_Liege_results = pd.concat([df_SVM_rbf_Liege_results, df], axis=0)

df_SVM_rbf_Liege_results.reset_index(inplace=True, drop=True)

# %%
column_names = df_SVM_rbf_Liege_results.columns.tolist()
average_cv_test_r2_column = [col for col in column_names if "Average CV Test R2" in col]

heatmap_data = df_SVM_rbf_Liege_results.pivot("Feature Combination", "Target", average_cv_test_r2_column[0])

# Generate the heatmap
# plt.figure()
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='RdBu_r', vmin=-0.4, vmax=0.4)
plt.title('Heatmap of Average CV Test R2, SVM-rbf, Liege dataset')
plt.tight_layout()
plt.show()
plt.close()

# %%
"""
XGBoost model results for all datasets
"""
model_list = ['Linear', 'Ridge', 'SVM-linear', 'SVM-rbf', 'rf', 'XGBoost', 'CV_AutoGluon']
all_feature_list = ['Sleep', 'Cov', 'Brain', 'CT', 'SA', 'Subcor', 'Sleep_Cov', 'Sleep_Cov_Brain', 'Sleep_Cov_CT',
                    'Sleep_Cov_SA', 'Sleep_Cov_Subcor', 'Sleep_Brain', 'Sleep_CT', 'Sleep_SA', 'Sleep_Subcor',
                    'Cov_Brain', 'Cov_CT', 'Cov_SA', 'Cov_Subcor', 'Sleep_Shuffle_Cov', 'Sleep_Cov_Subcor_Shuffle',
                    'Sleep_Shuffle_Cov_Subcor', 'Sleep_Shuffle_Subcor', 'Cov_Brain_Shuffle', 'Cov_Subcor_Shuffle']
sleep_cov_feature_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov']
sleep_cov_subcor_feature_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Subcor', 'Sleep_Cov_Subcor', 'Sleep_Cov_Subcor_Shuffle',
                                 'Sleep_Shuffle_Cov_Subcor', 'Sleep_Subcor', 'Sleep_Shuffle_Subcor']
cov_subcor_feature_list = ['Cov', 'Subcor', 'Cov_Subcor', 'Cov_Subcor_Shuffle']


def create_nan_df(columns):
    return pd.DataFrame({col: [np.nan] for col in columns})


# Process Stroop results
df_Stroop_results = None
for model in model_list:
    for feature in all_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Stroop', 'SHIP_Trend', feature, 'results.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
        else:
            print(f"Missing file for model: {model}, feature: {feature}")
            # Create a placeholder DataFrame with NaN values
            if df_Stroop_results is not None:
                df = create_nan_df(df_Stroop_results.columns)
            else:
                # If df_Stroop_results is None, we don't have columns yet, so create a single row with NaNs
                df = pd.DataFrame([[np.nan] * len(feature_list_Stroop)], columns=feature_list_Stroop)

        if df_Stroop_results is None:
            df_Stroop_results = df
        else:
            df_Stroop_results = pd.concat([df_Stroop_results, df], axis=0)

df_Stroop_results.reset_index(inplace=True, drop=True)

# Process Memory results
df_Memory_results = None
for model in model_list:
    for feature in all_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Memory', 'SHIP_Trend', feature, 'results.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
        else:
            print(f"Missing file for model: {model}, feature: {feature}")
            # Create a placeholder DataFrame with NaN values
            if df_Memory_results is not None:
                df = create_nan_df(df_Memory_results.columns)
            else:
                # If df_Memory_results is None, we don't have columns yet, so create a single row with NaNs
                df = pd.DataFrame([[np.nan] * len(feature_list_Memory)], columns=feature_list_Memory)

        if df_Memory_results is None:
            df_Memory_results = df
        else:
            df_Memory_results = pd.concat([df_Memory_results, df], axis=0)

df_Memory_results.reset_index(inplace=True, drop=True)

# %%
df_Stroop_XGBoost_results = df_Stroop_results[df_Stroop_results['Model'] == 'XGBoost']

# Set 'Feature Combination' as the index
df = df_Stroop_XGBoost_results.copy()
# Select only the relevant columns
df_subset = df[['Feature Combination', 'Average CV Test Pearson r', 'Performance Test Set Pearson r Liege']]

# Set 'Feature Combination' as the index
df_subset.set_index('Feature Combination', inplace=True)

# Transpose the DataFrame to have the desired format for the heatmap
df_subset = df_subset.T

# Create the heatmap
plt.figure(figsize=(16, 3))
cmap = 'Reds'
vmin, vmax = 0, 1
sns.heatmap(df_subset, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f")

# Add titles and labels
plt.title('Heatmap of Pearson r values')
plt.xlabel('Feature Combination')
plt.ylabel('Site')
# rotate the x-axis labels
plt.xticks(rotation=45)
# rotate the y-axis labels
plt.yticks(rotation=0)
plt.tight_layout()
# Show the plot
plt.show()
plt.close()


