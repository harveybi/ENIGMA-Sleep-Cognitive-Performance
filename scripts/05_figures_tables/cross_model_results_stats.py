import os

import starbars
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
"""
Make a heat map of the cross model results, x-axis is the Model, y-axis is the Input
"""
# model_list = ['Dummy', 'Linear', 'Ridge', 'SVM-linear', 'SVM-rbf', 'rf', 'XGBoost', 'JURECA_AutoGluon']
model_list = ['Linear', 'Ridge', 'SVM-linear', 'SVM-rbf', 'rf', 'XGBoost', 'CV_AutoGluon']
all_feature_list = ['Sleep', 'Cov', 'Brain', 'CT', 'SA', 'Subcor', 'Sleep_Cov', 'Sleep_Cov_Brain', 'Sleep_Cov_CT',
                    'Sleep_Cov_SA', 'Sleep_Cov_Subcor', 'Sleep_Brain', 'Sleep_CT', 'Sleep_SA', 'Sleep_Subcor',
                    'Cov_Brain', 'Cov_CT', 'Cov_SA', 'Cov_Subcor', 'Sleep_Shuffle_Cov', 'Sleep_Cov_Subcor_Shuffle',
                    'Sleep_Shuffle_Cov_Subcor', 'Sleep_Shuffle_Subcor', 'Cov_Brain_Shuffle', 'Cov_Subcor_Shuffle']
sleep_cov_feature_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov']
sleep_cov_subcor_feature_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Subcor', 'Sleep_Cov_Subcor', 'Sleep_Cov_Subcor_Shuffle',
                                 'Sleep_Shuffle_Cov_Subcor', 'Sleep_Subcor', 'Sleep_Shuffle_Subcor']
cov_subcor_feature_list = ['Cov', 'Subcor', 'Cov_Subcor', 'Cov_Subcor_Shuffle']

# feature_list_Stroop = ['Sleep', 'Cov', 'Sleep_Cov',
#                        'Sleep_Cov_Subcor', 'Cov_Subcor', 'Sleep_Shuffle_Cov',
#                        'Sleep_Cov_Subcor_Shuffle', 'Cov_Subcor_Shuffle']
# feature_list_Memory = ['Sleep', 'Cov', 'Sleep_Cov',
#                        'Sleep_Cov_Subcor', 'Cov_Subcor', 'Sleep_Shuffle_Cov',
#                        'Sleep_Cov_Subcor_Shuffle', 'Cov_Subcor_Shuffle']

# %%
# for loop to check which model missed which feature's results. Results path is like /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/rf/Stroop/SHIP_Trend/Cov_Subcor_Shuffle/results.csv
print('\nStroop')
for model in model_list:
    for feature in all_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Stroop', 'SHIP_Trend', feature, 'results.csv')
        if not os.path.exists(path):
            print(model, feature, 'missing')

print('\nMemory')
for model in model_list:
    for feature in all_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Memory', 'SHIP_Trend', feature, 'results.csv')
        if not os.path.exists(path):
            print(model, feature, 'missing')

# %%
"""
Heatmap for all features
"""


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
# model_order = ['AutoGluon', 'XGBoost', 'Random Forest', 'SVM-rbf', 'SVM-linear', 'Ridge Regression',
#                'Linear Regression', 'Dummy']
model_order = ['AutoGluon', 'XGBoost', 'Random Forest', 'SVM-rbf', 'SVM-linear', 'Ridge Regression',
               'Linear Regression']

# %%
# Stroop
# make a heat map of the results for different metrics. x-axis is the Model, y-axis is the Input
metrics = ['Average CV Test R2', 'Performance Test Set R2 Liege', 'Performance Test Set R2 Liege COF',
           'Performance Test Set R2 Liege COGNAP', 'Performance Test Set R2 Liege COF_COGNAP',
           'Average CV Test Pearson r', 'Performance Test Set Pearson r Liege',
           'Performance Test Set Pearson r Liege COF', 'Performance Test Set Pearson r Liege COGNAP',
           'Performance Test Set Pearson r Liege COF_COGNAP']

for metric in metrics:
    # Pivot the data and reorder columns based on model_order
    # pivoted_data = df_Stroop_results.pivot_table(index='Feature Combination', columns='Model', values=metric)
    pivoted_data = df_Stroop_results.pivot_table(index='Model', columns='Feature Combination', values=metric)
    # reordering the rows based on model_order
    pivoted_data = pivoted_data.reindex(index=model_order)
    # reordering the columns based on feature_list
    pivoted_data = pivoted_data.reindex(columns=all_feature_list)
    # pivoted_data = pivoted_data.reindex(columns=model_order)

    # Set cmap and value range based on the metric
    if 'R2' in metric:
        cmap = 'RdBu_r'
        vmin, vmax = -0.5, 0.5
    elif 'Pearson r' in metric:
        cmap = 'Reds'
        vmin, vmax = 0, 1

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(24, 8))
    sns.heatmap(pivoted_data, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f", ax=ax)
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.title(metric + ', Stroop')
    plt.tight_layout()
    plt.show()
    plt.close()

# %%
# Memory
metrics = ['Average CV Test R2', 'Performance Test Set R2 Liege', 'Performance Test Set R2 Liege COF',
           'Performance Test Set R2 Liege COGNAP', 'Performance Test Set R2 Liege COF_COGNAP',
           'Performance Test Set R2 KI',
           'Average CV Test Pearson r', 'Performance Test Set Pearson r Liege',
           'Performance Test Set Pearson r Liege COF', 'Performance Test Set Pearson r Liege COGNAP',
           'Performance Test Set Pearson r Liege COF_COGNAP', 'Performance Test Set Pearson r KI']

for metric in metrics:
    # Pivot the data and reorder columns based on model_order
    pivoted_data = df_Memory_results.pivot_table(index='Model', columns='Feature Combination', values=metric)
    pivoted_data = pivoted_data.reindex(index=model_order)
    pivoted_data = pivoted_data.reindex(columns=all_feature_list)

    # Set cmap and value range based on the metric
    if 'R2' in metric:
        cmap = 'RdBu_r'
        vmin, vmax = -0.5, 0.5
    elif 'Pearson r' in metric:
        cmap = 'Reds'
        vmin, vmax = 0, 1

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(24, 8))
    sns.heatmap(pivoted_data, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f", ax=ax)
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.title(metric + ', Memory')
    plt.tight_layout()
    plt.show()
    plt.close()

# %%
"""
Heatmap for sleep_cov_feature_list
"""
# Process Stroop results
df_Stroop_results = None
for model in model_list:
    for feature in sleep_cov_feature_list:
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
    for feature in sleep_cov_feature_list:
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
# Stroop
# make a heat map of the results for different metrics. x-axis is the Model, y-axis is the Input
# metrics = ['Average CV Test R2', 'Performance Test Set R2 Liege', 'Performance Test Set R2 Liege COF',
#            'Performance Test Set R2 Liege COGNAP', 'Performance Test Set R2 Liege COF_COGNAP',
#            'Average CV Test Pearson r', 'Performance Test Set Pearson r Liege',
#            'Performance Test Set Pearson r Liege COF', 'Performance Test Set Pearson r Liege COGNAP',
#            'Performance Test Set Pearson r Liege COF_COGNAP']
metrics = ['Average CV Test R2', 'Average CV Test Pearson r', 'Performance Test Set R2 Liege', 'Performance Test Set Pearson r Liege']

for metric in metrics:
    # Pivot the data and reorder columns based on model_order
    # pivoted_data = df_Stroop_results.pivot_table(index='Feature Combination', columns='Model', values=metric)
    pivoted_data = df_Stroop_results.pivot_table(index='Model', columns='Feature Combination', values=metric)
    # reordering the rows based on model_order
    pivoted_data = pivoted_data.reindex(index=model_order)
    # reordering the columns based on feature_list
    pivoted_data = pivoted_data.reindex(columns=sleep_cov_feature_list)
    # pivoted_data = pivoted_data.reindex(columns=model_order)

    # Set cmap and value range based on the metric
    if 'R2' in metric:
        cmap = 'RdBu_r'
        vmin, vmax = -0.5, 0.5
    elif 'Pearson r' in metric:
        cmap = 'Reds'
        vmin, vmax = 0, 1

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivoted_data, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f", ax=ax)
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.title(metric + ', Stroop')
    plt.tight_layout()
    plt.show()
    plt.close()

# %%
# Memory
# metrics = ['Average CV Test R2', 'Performance Test Set R2 Liege', 'Performance Test Set R2 Liege COF',
#            'Performance Test Set R2 Liege COGNAP', 'Performance Test Set R2 Liege COF_COGNAP',
#            'Performance Test Set R2 KI',
#            'Average CV Test Pearson r', 'Performance Test Set Pearson r Liege',
#            'Performance Test Set Pearson r Liege COF', 'Performance Test Set Pearson r Liege COGNAP',
#            'Performance Test Set Pearson r Liege COF_COGNAP', 'Performance Test Set Pearson r KI']
metrics = ['Average CV Test R2', 'Average CV Test Pearson r',
           'Performance Test Set R2 Liege', 'Performance Test Set R2 KI',
           'Performance Test Set Pearson r Liege', 'Performance Test Set Pearson r KI']

for metric in metrics:
    # Pivot the data and reorder columns based on model_order
    pivoted_data = df_Memory_results.pivot_table(index='Model', columns='Feature Combination', values=metric)
    pivoted_data = pivoted_data.reindex(index=model_order)
    pivoted_data = pivoted_data.reindex(columns=sleep_cov_feature_list)

    # Set cmap and value range based on the metric
    if 'R2' in metric:
        cmap = 'RdBu_r'
        vmin, vmax = -0.5, 0.5
    elif 'Pearson r' in metric:
        cmap = 'Reds'
        vmin, vmax = 0, 1

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivoted_data, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f", ax=ax)
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.title(metric + ', Memory')
    plt.tight_layout()
    plt.show()
    plt.close()

# %%
"""
Heatmap for sleep_cov_subcor_feature_list
"""
# Process Stroop results
df_Stroop_results = None
for model in model_list:
    for feature in sleep_cov_subcor_feature_list:
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
    for feature in sleep_cov_subcor_feature_list:
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
# Stroop
# make a heat map of the results for different metrics. x-axis is the Model, y-axis is the Input
metrics = ['Average CV Test R2']
# metrics = ['Average CV Test R2', 'Performance Test Set R2 Liege', 'Performance Test Set R2 Liege COF',
#            'Performance Test Set R2 Liege COGNAP', 'Performance Test Set R2 Liege COF_COGNAP',
#            'Average CV Test Pearson r', 'Performance Test Set Pearson r Liege',
#            'Performance Test Set Pearson r Liege COF', 'Performance Test Set Pearson r Liege COGNAP',
#            'Performance Test Set Pearson r Liege COF_COGNAP']

for metric in metrics:
    # Pivot the data and reorder columns based on model_order
    # pivoted_data = df_Stroop_results.pivot_table(index='Feature Combination', columns='Model', values=metric)
    pivoted_data = df_Stroop_results.pivot_table(index='Model', columns='Feature Combination', values=metric)
    # reordering the rows based on model_order
    pivoted_data = pivoted_data.reindex(index=model_order)
    # reordering the columns based on feature_list
    pivoted_data = pivoted_data.reindex(columns=sleep_cov_subcor_feature_list)
    # pivoted_data = pivoted_data.reindex(columns=model_order)

    # Set cmap and value range based on the metric
    if 'R2' in metric:
        cmap = 'RdBu_r'
        vmin, vmax = -0.5, 0.5
    elif 'Pearson r' in metric:
        cmap = 'Reds'
        vmin, vmax = 0, 1

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivoted_data, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f", ax=ax)
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.title(metric + ', Stroop')
    plt.tight_layout()
    plt.show()
    plt.close()

# %%
# Memory
metrics = ['Average CV Test R2']
# metrics = ['Average CV Test R2', 'Performance Test Set R2 Liege', 'Performance Test Set R2 Liege COF',
#            'Performance Test Set R2 Liege COGNAP', 'Performance Test Set R2 Liege COF_COGNAP',
#            'Performance Test Set R2 KI',
#            'Average CV Test Pearson r', 'Performance Test Set Pearson r Liege',
#            'Performance Test Set Pearson r Liege COF', 'Performance Test Set Pearson r Liege COGNAP',
#            'Performance Test Set Pearson r Liege COF_COGNAP', 'Performance Test Set Pearson r KI']

for metric in metrics:
    # Pivot the data and reorder columns based on model_order
    pivoted_data = df_Memory_results.pivot_table(index='Model', columns='Feature Combination', values=metric)
    pivoted_data = pivoted_data.reindex(index=model_order)
    pivoted_data = pivoted_data.reindex(columns=sleep_cov_subcor_feature_list)

    # Set cmap and value range based on the metric
    if 'R2' in metric:
        cmap = 'RdBu_r'
        vmin, vmax = -0.5, 0.5
    elif 'Pearson r' in metric:
        cmap = 'Reds'
        vmin, vmax = 0, 1

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivoted_data, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f", ax=ax)
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.title(metric + ', Memory')
    plt.tight_layout()
    plt.show()
    plt.close()
