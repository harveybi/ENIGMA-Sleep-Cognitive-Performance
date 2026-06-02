import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')
import utils

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import julearn
from julearn.stats.corrected_ttest import corrected_ttest
import itertools
import starbars

from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
)

# %%
"""
Barplot to compare different model performances for Sleep_Cov_Brain
"""
model_list = ['Linear', 'Ridge', 'SVM-linear', 'SVM-rbf', 'rf', 'XGBoost', 'AutoGluon']  # for boxplot, dummy model is not included to compare

cross_model_feature_list = ['Cov', 'Sleep', 'Sleep_Cov', 'Brain', 'Sleep_Cov_Brain', 'Sleep_Shuffle_Cov']

target_list = ['Stroop', 'Memory']

# %%
print('\nStroop')
for model in model_list:
    for feature in all_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Stroop', 'SHIP_Trend', feature, 'scores.csv')
        if not os.path.exists(path):
            print(model, feature, 'missing')

print('\nMemory')
for model in model_list:
    for feature in all_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Memory', 'SHIP_Trend', feature, 'scores.csv')
        if not os.path.exists(path):
            print(model, feature, 'missing')

# %%
# load scores.csv for each models separately.
feature = 'Sleep_Cov_Brain'
# feature = 'Sleep_Cov'

df_Stroop_Linear = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/Linear/Stroop/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Stroop_Ridge = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/Ridge/Stroop/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Stroop_SVM_linear = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/SVM-linear/Stroop/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Stroop_SVM_rbf = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/SVM-rbf/Stroop/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Stroop_rf = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/rf/Stroop/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Stroop_XGBoost = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Stroop_CV_AutoGluon = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Stroop/SHIP_Trend/' + feature + '/scores.csv'
)

df_Memory_Linear = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/Linear/Memory/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Memory_Ridge = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/Ridge/Memory/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Memory_SVM_linear = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/SVM-linear/Memory/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Memory_SVM_rbf = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/SVM-rbf/Memory/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Memory_rf = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/rf/Memory/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Memory_XGBoost = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Memory/SHIP_Trend/' + feature + '/scores.csv',
    index_col=0)
df_Memory_CV_AutoGluon = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Memory/SHIP_Trend/' + feature + '/scores.csv'
)

# %%
"""
Use the simple correct t-test to compare the performance of different models. Compare 'test_r_corr' only.
"""
# Dictionary of models and their entire dataframes
stroop_models_dict = {
    # 'Dummy': df_Stroop_Dummy,
    'Linear': df_Stroop_Linear,
    'Ridge': df_Stroop_Ridge,
    'SVM-linear': df_Stroop_SVM_linear,
    'SVM-rbf': df_Stroop_SVM_rbf,
    'rf': df_Stroop_rf,
    'XGBoost': df_Stroop_XGBoost,
    'AutoGluon': df_Stroop_CV_AutoGluon
}

memory_models_dict = {
    # 'Dummy': df_Memory_Dummy,
    'Linear': df_Memory_Linear,
    'Ridge': df_Memory_Ridge,
    'SVM-linear': df_Memory_SVM_linear,
    'SVM-rbf': df_Memory_SVM_rbf,
    'rf': df_Memory_rf,
    'XGBoost': df_Memory_XGBoost,
    'AutoGluon': df_Memory_CV_AutoGluon
}

# %%
def stats_simple_t_test(models_dict):
    # Perform pairwise t-tests for 'test_r_corr' column
    results = []
    p_values = []

    for (model1, df1), (model2, df2) in itertools.combinations(models_dict.items(), 2):
        # Extract 'test_r_corr' column
        test_r_corr1 = df1['test_r_corr']
        test_r_corr2 = df2['test_r_corr']

        # Paired t-test
        t_stat, p_value = ttest_rel(test_r_corr1, test_r_corr2)

        # Store the results
        results.append({
            'Model 1': model1,
            'Model 2': model2,
            't-statistic': t_stat,
            'p-value': p_value
        })

        # Collect p-values for multiple comparison correction
        p_values.append(p_value)

    # Apply Bonferroni correction
    _, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')

    # Add corrected p-values to results
    for i, corrected_p_value in enumerate(corrected_p_values):
        results[i]['corrected p-value (Bonferroni)'] = corrected_p_value

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # based on results_df's 'corrected p-value (Bonferroni)' column, add a new column 'significant' to results_df. If 'corrected p-value (Bonferroni)' < 0.05, 'significant' is True, otherwise False.
    results_df['significant'] = results_df['corrected p-value (Bonferroni)'] < 0.05

    return results_df

# Perform t-tests for Stroop
stats_stroop_df_0 = stats_simple_t_test(stroop_models_dict)
# Perform t-tests for Memory
stats_memory_df_0 = stats_simple_t_test(memory_models_dict)

# %%
"""
Use the julearn correct t-test to compare the performance of different models. Compare 'test_r_corr' only.
Case 1: Use the cv_mdsum from df_Stroop_rf as the cv_mdsum for all models. The train set and test set sizes are different.
"""
# for df_Stroop_XGBoost, sort the rows by 'repeat' and 'fold' columns.
df_Stroop_XGBoost = df_Stroop_XGBoost.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
# for df_Memory_XGBoost, sort the rows by 'repeat' and 'fold' columns.
df_Memory_XGBoost = df_Memory_XGBoost.sort_values(by=['repeat', 'fold']).reset_index(drop=True)

# %%
# load data and generate cv used for XGBoost and AutoGluon
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')

# convert sleep measurements units
sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']
df_SHIP_ml = utils.convert_units(df_SHIP, sleep_dur_cols, sleep_eff_cols)

# add a column of 'Age_Group' after 'Age_at_Scan'.
df_SHIP_ml = utils.add_age_groups(df_SHIP_ml)
# add a column of Group which represents the group of age groups and SEX groups
df_SHIP_ml = utils.add_groups_age_sex(df_SHIP_ml)

def generate_kfold(df_train, y=None, n_splits=5, random_state=0, stratified=False, n_repeats=1):
    # kf = None
    X_data = df_train
    y_data = df_train[y] if y is not None else None

    if stratified and (y is not None):
        if n_repeats > 1:
            kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kf.get_n_splits(X_data, y_data)
        return [[train_index, test_index] for train_index, test_index in kf.split(X_data, y_data)]

    else:
        if n_repeats > 1:
            kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kf.get_n_splits(X_data)
        return [[train_index, test_index] for train_index, test_index in kf.split(X_data)]


cv_splitter = generate_kfold(df_SHIP_ml, y='Group_Age_SEX', n_splits=5, random_state=42, stratified=True, n_repeats=10)

n_train = []
n_test = []
for train_index, test_index in cv_splitter:
    n_train.append(len(train_index))
    n_test.append(len(test_index))

# %%
# add two columns to df_Stroop_XGBoost, df_Memory_XGBoost, df_Stroop_CV_AutoGluon, df_Memory_CV_AutoGluon. Columns are 'n_train' and 'n_test'.
df_Stroop_XGBoost['n_train'] = n_train
df_Stroop_XGBoost['n_test'] = n_test
df_Memory_XGBoost['n_train'] = n_train
df_Memory_XGBoost['n_test'] = n_test
df_Stroop_CV_AutoGluon['n_train'] = n_train
df_Stroop_CV_AutoGluon['n_test'] = n_test
df_Memory_CV_AutoGluon['n_train'] = n_train
df_Memory_CV_AutoGluon['n_test'] = n_test

df_Stroop_XGBoost['model'] = 'XGBoost'
df_Memory_XGBoost['model'] = 'XGBoost'
df_Stroop_CV_AutoGluon['model'] = 'AutoGluon'
df_Memory_CV_AutoGluon['model'] = 'AutoGluon'

# add 'model' column to other dataframes. Values are 'Linear Regression', 'Ridge Regression', 'SVM-linear', 'SVM-rbf', 'Random Forest'.
df_Stroop_Linear['model'] = 'Linear Regression'
df_Stroop_Ridge['model'] = 'Ridge Regression'
df_Stroop_SVM_linear['model'] = 'SVM-linear'
df_Stroop_SVM_rbf['model'] = 'SVM-rbf'
df_Stroop_rf['model'] = 'Random Forest'

df_Memory_Linear['model'] = 'Linear Regression'
df_Memory_Ridge['model'] = 'Ridge Regression'
df_Memory_SVM_linear['model'] = 'SVM-linear'
df_Memory_SVM_rbf['model'] = 'SVM-rbf'
df_Memory_rf['model'] = 'Random Forest'

# %%
cv_mdsum = df_Stroop_Ridge['cv_mdsum'][0]

df_Stroop_XGBoost['cv_mdsum'] = cv_mdsum
df_Memory_XGBoost['cv_mdsum'] = cv_mdsum

df_Stroop_CV_AutoGluon['cv_mdsum'] = cv_mdsum
df_Memory_CV_AutoGluon['cv_mdsum'] = cv_mdsum

# %%
df_Stroop_XGBoost['train_r2'] = 0.0
df_Stroop_XGBoost['train_neg_mean_absolute_error'] = 0.0
df_Stroop_XGBoost['train_neg_root_mean_squared_error'] = 0.0
df_Stroop_XGBoost['train_r_corr'] = 0.0
df_Stroop_XGBoost['train_spearmanr'] = 0.0
df_Memory_XGBoost['train_r2'] = 0.0
df_Memory_XGBoost['train_neg_mean_absolute_error'] = 0.0
df_Memory_XGBoost['train_neg_root_mean_squared_error'] = 0.0
df_Memory_XGBoost['train_r_corr'] = 0.0
df_Memory_XGBoost['train_spearmanr'] = 0.0

df_Stroop_CV_AutoGluon['train_r2'] = 0.0
df_Stroop_CV_AutoGluon['train_neg_mean_absolute_error'] = 0.0
df_Stroop_CV_AutoGluon['train_neg_root_mean_squared_error'] = 0.0
df_Stroop_CV_AutoGluon['train_r_corr'] = 0.0
df_Stroop_CV_AutoGluon['train_spearmanr'] = 0.0
df_Memory_CV_AutoGluon['train_r2'] = 0.0
df_Memory_CV_AutoGluon['train_neg_mean_absolute_error'] = 0.0
df_Memory_CV_AutoGluon['train_neg_root_mean_squared_error'] = 0.0
df_Memory_CV_AutoGluon['train_r_corr'] = 0.0
df_Memory_CV_AutoGluon['train_spearmanr'] = 0.0

# %%
# for 'fold' column in df_Stroop_XGBoost, df_Memory_XGBoost, df_Stroop_CV_AutoGluon, df_Memory_CV_AutoGluon, replace the value of each row by 'repeat' value * 'fold' value - 1.
df_Stroop_XGBoost['fold'] = df_Stroop_XGBoost['fold'] + (5 * (df_Stroop_XGBoost['repeat'] - 1)) - 1
df_Memory_XGBoost['fold'] = df_Memory_XGBoost['fold'] + (5 * (df_Memory_XGBoost['repeat'] - 1)) - 1
df_Stroop_CV_AutoGluon['fold'] = df_Stroop_CV_AutoGluon['fold'] + (5 * (df_Stroop_CV_AutoGluon['repeat'] - 1)) - 1
df_Memory_CV_AutoGluon['fold'] = df_Memory_CV_AutoGluon['fold'] + (5 * (df_Memory_CV_AutoGluon['repeat'] - 1)) - 1

df_Stroop_XGBoost['repeat'] = 0
df_Memory_XGBoost['repeat'] = 0
df_Stroop_CV_AutoGluon['repeat'] = 0
df_Memory_CV_AutoGluon['repeat'] = 0

# %%
from julearn.stats.corrected_ttest import corrected_ttest

stats_stroop_df_1 = corrected_ttest(df_Stroop_Linear, df_Stroop_Ridge, df_Stroop_SVM_linear, df_Stroop_SVM_rbf,
                                    df_Stroop_rf, df_Stroop_XGBoost, df_Stroop_CV_AutoGluon)
stats_memory_df_1 = corrected_ttest(df_Memory_Linear, df_Memory_Ridge, df_Memory_SVM_linear, df_Memory_SVM_rbf,
                                    df_Memory_rf, df_Memory_XGBoost, df_Memory_CV_AutoGluon)

# %%
stats_stroop_df_1['significance'] = stats_stroop_df_1['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')
stats_memory_df_1['significance'] = stats_memory_df_1['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')

# %%
# only keep rows which 'metric' is 'test_r_corr'
stats_stroop_df_1_corr = stats_stroop_df_1[stats_stroop_df_1['metric'] == 'test_r_corr']
stats_memory_df_1_corr = stats_memory_df_1[stats_memory_df_1['metric'] == 'test_r_corr']

# %%
# for the metric column in stats_stroop_df, only keep the rows which values are test_r_corr and test_r2.
stats_stroop_df_1_cld = stats_stroop_df_1[stats_stroop_df_1['metric'].isin(['test_r_corr', 'test_r2'])]
# add a new column 'significant' to stats_stroop_df_cld. If 'p-val-corrected' < 0.05, 'significant' is True, otherwise False.
stats_stroop_df_1_cld['significant'] = stats_stroop_df_1_cld['p-val-corrected'] < 0.05

stats_memory_df_1_cld = stats_memory_df_1[stats_memory_df_1['metric'].isin(['test_r_corr', 'test_r2'])]
# add a new column 'significant' to stats_memory_df_cld. If 'p-val-corrected' < 0.05, 'significant' is True, otherwise False.
stats_memory_df_1_cld['significant'] = stats_memory_df_1_cld['p-val-corrected'] < 0.05

# %%
"""
Use the julearn correct t-test to compare the performance of different models. Compare 'test_r_corr' only.
Case 2: Use the cv_mdsum from df_Stroop_rf as the cv_mdsum for all models, and the test and train set size are also the same.
"""
from sklearn.model_selection import (
    check_cv,
    cross_validate,
)
from julearn.utils import _compute_cvmdsum

problem_type = "regression"

cv_outer = check_cv(
        cv_splitter,  # type: ignore
        classifier=problem_type == "classification",
    )

cv_mdsum = _compute_cvmdsum(cv_outer)

# %%
# since cv_mdsum should be the same for all models, let's use XGBoost's cv_mdsum as the cv_mdsum for all models.
df_Stroop_Linear['cv_mdsum'] = cv_mdsum
df_Stroop_Ridge['cv_mdsum'] = cv_mdsum
df_Stroop_SVM_linear['cv_mdsum'] = cv_mdsum
df_Stroop_SVM_rbf['cv_mdsum'] = cv_mdsum
df_Stroop_rf['cv_mdsum'] = cv_mdsum
df_Stroop_XGBoost['cv_mdsum'] = cv_mdsum
df_Stroop_CV_AutoGluon['cv_mdsum'] = cv_mdsum

df_Memory_Linear['cv_mdsum'] = cv_mdsum
df_Memory_Ridge['cv_mdsum'] = cv_mdsum
df_Memory_SVM_linear['cv_mdsum'] = cv_mdsum
df_Memory_SVM_rbf['cv_mdsum'] = cv_mdsum
df_Memory_rf['cv_mdsum'] = cv_mdsum
df_Memory_XGBoost['cv_mdsum'] = cv_mdsum
df_Memory_CV_AutoGluon['cv_mdsum'] = cv_mdsum

# %%
stats_stroop_df_2 = corrected_ttest(df_Stroop_Linear, df_Stroop_Ridge, df_Stroop_SVM_linear, df_Stroop_SVM_rbf,
                                    df_Stroop_rf, df_Stroop_XGBoost, df_Stroop_CV_AutoGluon)
stats_memory_df_2 = corrected_ttest(df_Memory_Linear, df_Memory_Ridge, df_Memory_SVM_linear, df_Memory_SVM_rbf,
                                    df_Memory_rf, df_Memory_XGBoost, df_Memory_CV_AutoGluon)

stats_stroop_df_2_cld = stats_stroop_df_2[stats_stroop_df_2['metric'].isin(['test_r_corr', 'test_r2'])]
stats_stroop_df_2_cld['significant'] = stats_stroop_df_2_cld['p-val-corrected'] < 0.05

stats_memory_df_2_cld = stats_memory_df_2[stats_memory_df_2['metric'].isin(['test_r_corr', 'test_r2'])]
stats_memory_df_2_cld['significant'] = stats_memory_df_2_cld['p-val-corrected'] < 0.05

# %%
"""
Make boxplot for Stroop and Memory
"""
df_Stroop = pd.concat([df_Stroop_Linear, df_Stroop_Ridge, df_Stroop_SVM_linear, df_Stroop_SVM_rbf, df_Stroop_rf, df_Stroop_XGBoost, df_Stroop_CV_AutoGluon])
df_Memory = pd.concat([df_Memory_Linear, df_Memory_Ridge, df_Memory_SVM_linear, df_Memory_SVM_rbf, df_Memory_rf, df_Memory_XGBoost, df_Memory_CV_AutoGluon])

# %%
"""
Plot of Pearson correlation coefficient for Stroop
"""
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Order of models to display on x-axis
order = ['Linear Regression', 'Ridge Regression', 'SVM-linear', 'SVM-rbf', 'Random Forest', 'XGBoost', 'AutoGluon']

# Create a figure with two subplots in a row (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)  # sharey=True ensures same y-axis scale

# Boxplot for Stroop (no color, only outlines)
sns.boxplot(data=df_Stroop, x='model', y='test_r_corr', order=order, ax=axes[0], width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
# Swarmplot on top of the boxplot for Stroop
sns.swarmplot(data=df_Stroop, x='model', y='test_r_corr', order=order, ax=axes[0],
              palette="Set2", size=4)

axes[0].set_title('Stroop')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Pearson Correlation Coefficient')

# Boxplot for Memory (no color, only outlines)
sns.boxplot(data=df_Memory, x='model', y='test_r_corr', order=order, ax=axes[1], width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
# Swarmplot on top of the boxplot for Memory
sns.swarmplot(data=df_Memory, x='model', y='test_r_corr', order=order, ax=axes[1],
              palette="Set2", size=4)

axes[1].set_title('Memory')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.15, 0.60)  # Adjust the range according to your data

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=45)

# Adjust layout to ensure no overlap
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

# Order of models to display on x-axis
order = ['Linear Regression', 'Ridge Regression', 'SVM-linear', 'SVM-rbf', 'Random Forest', 'XGBoost', 'AutoGluon']

# Create a figure with two subplots in a row (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)  # sharey=True ensures same y-axis scale

# Boxplot for Stroop (no color, only outlines)
sns.boxplot(data=df_Stroop, x='model', y='test_r_corr', order=order, ax=axes[0], width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
# Swarmplot on top of the boxplot for Stroop
sns.swarmplot(data=df_Stroop, x='model', y='test_r_corr', order=order, ax=axes[0],
              palette="Set2", size=4)

axes[0].set_title('Stroop', fontsize=18)  # Set title font size
# axes[0].set_xlabel('Model', fontsize=14)  # Set x-axis label font size
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=18)  # Set y-axis label font size

# Boxplot for Memory (no color, only outlines)
sns.boxplot(data=df_Memory, x='model', y='test_r_corr', order=order, ax=axes[1], width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
# Swarmplot on top of the boxplot for Memory
sns.swarmplot(data=df_Memory, x='model', y='test_r_corr', order=order, ax=axes[1],
              palette="Set2", size=4)

axes[1].set_title('Memory', fontsize=18)  # Set title font size
# axes[1].set_xlabel('Model', fontsize=14)  # Set x-axis label font size
axes[1].set_xlabel('')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.15, 0.60)  # Adjust the range according to your data

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=45, labelsize=16)  # Set x-axis tick label size
    ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Adjust layout to ensure no overlap
plt.tight_layout()

# save as svg
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'Stroop_Memory_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')
# Show the plot
plt.show()
plt.close()

# %%
"""
Borplot
"""
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

# Order of models to display on x-axis
order = ['Linear Regression', 'Ridge Regression', 'SVM-linear', 'SVM-rbf', 'Random Forest', 'XGBoost', 'AutoGluon']

# Create a figure with two subplots in a row (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)  # sharey=True ensures same y-axis scale

# Function to calculate outliers
def get_outliers(data, column):
    """Returns data points that are outliers based on 1.5*IQR rule."""
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Barplot for Stroop with built-in error bars
sns.barplot(
    data=df_Stroop, x='model', y='test_r_corr', order=order, ax=axes[0],
    ci='sd',  # Use standard deviation for error bars
    palette="Set2"
)

# Get and plot outliers for Stroop
outliers_stroop = get_outliers(df_Stroop, 'test_r_corr')
axes[0].scatter(
    x=[order.index(model) for model in outliers_stroop['model']],
    y=outliers_stroop['test_r_corr'],
    color='black', alpha=0.8, s=30, label='Outliers'
)

axes[0].set_title('Stroop', fontsize=18)
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=18)

# Barplot for Memory with built-in error bars
sns.barplot(
    data=df_Memory, x='model', y='test_r_corr', order=order, ax=axes[1],
    ci='sd',  # Use standard deviation for error bars
    palette="Set2"
)

# Get and plot outliers for Memory
outliers_memory = get_outliers(df_Memory, 'test_r_corr')
axes[1].scatter(
    x=[order.index(model) for model in outliers_memory['model']],
    y=outliers_memory['test_r_corr'],
    color='black', alpha=0.8, s=30, label='Outliers'
)

axes[1].set_title('Memory', fontsize=18)
axes[1].set_xlabel('')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.15, 0.60)

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=18)

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'Stroop_Memory_Pearson_corr_barplot_outliers.svg', format='svg', dpi=1200, bbox_inches='tight')

# Show the plot
plt.show()
plt.close()

# %%----------------- SEPARATE COLOR BAR FIGURE -----------------
# Your color palette from the swarmplot
palette = sns.color_palette("Set2", len(order))

# Create a new figure for the color bar
fig_colorbar, ax_colorbar = plt.subplots(figsize=(6, 2))

# Create a colormap from the same palette used in the swarmplot
cmap = sns.color_palette(palette, as_cmap=True)

# Create a color bar using ScalarMappable with the updated cmap call
norm = plt.Normalize(vmin=0, vmax=len(order) - 1)
sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("Set2", len(order)), norm=norm)

# Add the color bar to the figure
cbar = fig_colorbar.colorbar(sm, cax=ax_colorbar, orientation="horizontal")
cbar.set_ticks(np.arange(len(order)))
cbar.set_ticklabels(order)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
# Show the color bar figure
plt.show()
plt.close()

# %%
pvalues = []
for pair in pairs:
    pvalue = stats_stroop_df.loc[
        ((stats_stroop_df['model_1'] == pair[0]) & (stats_stroop_df['model_2'] == pair[1])) |
        ((stats_stroop_df['model_1'] == pair[1]) & (stats_stroop_df['model_2'] == pair[0])),
        'p-val'
    ]

    if not pvalue.empty:
        pvalues.append(pvalue.values[0])
    else:
        print(f"No p-value found for pair: {pair}")
        pvalues.append(None)  # or some default value, e.g., np.nan

print("P-values:", pvalues)

formatted_pvalues = [f"p={p:.2e}" for p in pvalues]

formatted_star = []
for p in pvalues:
    if p > 0.05:
        formatted_star.append('ns')
    elif p > 0.01:
        formatted_star.append('*')
    elif p > 0.001:
        formatted_star.append('**')
    elif p > 0.0001:
        formatted_star.append('***')
    else:
        formatted_star.append('****')

# Plot with seaborn
# set figure size
# fig, ax = plt.figure(figsize=(7, 6))
fig, ax = plt.subplots(figsize=(8, 6))

# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="whitegrid", rc=custom_params)
sns.set_theme(style="whitegrid")

sns.swarmplot(
        x=x, y=y, color='gray',
        data=data, order=order,
        ax=ax, alpha=.5, size=5
    )

# ax = sns.boxplot(data=data, x=x, y=y, order=order, width=.5)
sns.boxplot(
        x=x, y=y, data=data,
        ax=ax,
        whis=[2.5, 97.5],
        color='w', zorder=1,
        showfliers=False,
    )

# Add annotations
annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
annotator.set_custom_annotations(formatted_star)
annotator.annotate()

# Label and show
# y-axis range from -0.25 to 0.3
# ax.set_ylim(-0.05, 0.8)
# change the value of x-axis
# plt.xticks(np.arange(4), order, rotation=45)
# ax.xaxis.set_tick_params(rotation=45)
# # name x-axis
# plt.xlabel('Model input of feature combination')
# # name y-axis
# plt.ylabel('Cross-validated average Pearson correlation coefficient')
# rename the y-axis by ax
ax.set_ylabel('Pearson correlation coefficient')
# title
ax.set_title('Stroop Prediction in SHIP-Trend')
fig.tight_layout()
# plt.savefig("./plot1A.png", bbox_inches='tight')
fig.show()
plt.close(fig)

# %%
"""
Plot of Pearson correlation coefficient for Memory
"""
data = df_Memory
x = 'model'
y = 'test_r_corr'
order = ['Ridge', 'SVM-rbf', 'XGBoost']

pairs = [('Ridge', 'SVM-rbf'),
         ('SVM-rbf', 'XGBoost')]
# pairs_raw = [('Cov', 'Sleep_Cov'), ('Sleep_Cov', 'Sleep_Shuffle_Cov'), ('Cov', 'Sleep_Shuffle_Cov')]

pvalues = []
for pair in pairs:
    pvalue = stats_stroop_df.loc[
        ((stats_stroop_df['model_1'] == pair[0]) & (stats_stroop_df['model_2'] == pair[1])) |
        ((stats_stroop_df['model_1'] == pair[1]) & (stats_stroop_df['model_2'] == pair[0])),
        'p-val'
    ]

    if not pvalue.empty:
        pvalues.append(pvalue.values[0])
    else:
        print(f"No p-value found for pair: {pair}")
        pvalues.append(None)  # or some default value, e.g., np.nan

print("P-values:", pvalues)

formatted_pvalues = [f"p={p:.2e}" for p in pvalues]

formatted_star = []
for p in pvalues:
    if p > 0.05:
        formatted_star.append('ns')
    elif p > 0.01:
        formatted_star.append('*')
    elif p > 0.001:
        formatted_star.append('**')
    elif p > 0.0001:
        formatted_star.append('***')
    else:
        formatted_star.append('****')

# Plot with seaborn
# set figure size
# fig, ax = plt.figure(figsize=(7, 6))
fig, ax = plt.subplots(figsize=(8, 6))

# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="whitegrid", rc=custom_params)
sns.set_theme(style="whitegrid")

sns.swarmplot(
        x=x, y=y, color='gray',
        data=data, order=order,
        ax=ax, alpha=.5, size=5
    )

# ax = sns.boxplot(data=data, x=x, y=y, order=order, width=.5)
sns.boxplot(
        x=x, y=y, data=data,
        ax=ax,
        whis=[2.5, 97.5],
        color='w', zorder=1,
        showfliers=False,
    )

# Add annotations
annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
annotator.set_custom_annotations(formatted_star)
annotator.annotate()

# Label and show
# y-axis range from -0.25 to 0.3
# ax.set_ylim(-0.05, 0.8)
# change the value of x-axis
# plt.xticks(np.arange(4), order, rotation=45)
# ax.xaxis.set_tick_params(rotation=45)
# # name x-axis
# plt.xlabel('Model input of feature combination')
# # name y-axis
# plt.ylabel('Cross-validated average Pearson correlation coefficient')
# rename the y-axis by ax
ax.set_ylabel('Pearson correlation coefficient')
# title
ax.set_title('Memory Prediction in SHIP-Trend')
fig.tight_layout()
# plt.savefig("./plot1A.png", bbox_inches='tight')
fig.show()
plt.close(fig)

# %%
# just boxplot for Stroop
# plt.figure(figsize=(10, 6))
sns.boxplot(data=df_Stroop, x='model', y='test_r2', width=.5)
# change the value of x-axis
plt.xticks(np.arange(5), ['Linear regression', 'Ridge regression', 'SVM-linear', 'SVM-rbf', 'XGBoost'], rotation=45)
# name x-axis
plt.xlabel('Machine learning models')
# name y-axis
plt.ylabel('Cross-validated average $R^2$')
plt.tight_layout()
plt.show()
plt.close()

# %%
df_Stroop_1 = pd.concat([df_Stroop_SVM_linear, df_Stroop_SVM_rbf, df_Stroop_XGBoost])  # , df_Stroop_CV_AutoGluon

# %%
sns.boxplot(data=df_Stroop_1, x='model', y='test_r2', width=.5)
# change the value of x-axis
plt.xticks(np.arange(3), ['SVM-linear', 'SVM-rbf', 'XGBoost'], rotation=45)
# name x-axis
plt.xlabel('Machine learning models')
# name y-axis
plt.ylabel('Cross-validated average $R^2$')
plt.tight_layout()
plt.show()
plt.close()

# %%
from statannotations.Annotator import Annotator

data = df_Stroop_1
x = 'model'
y = 'test_r2'
order = ['SVM-linear', 'SVM-rbf', 'XGBoost']

pairs = [('SVM-linear', 'SVM-rbf'),
         ('SVM-rbf', 'XGBoost')]

pvalues = [3.104254e-03, 6.571548e-02]

formatted_pvalues = [f"p={p:.2e}" for p in pvalues]

# creat formatted_star based on pvalues: p-value annotation legend:
#       ns: 5.00e-02 < p <= 1.00e+00
#        *: 1.00e-02 < p <= 5.00e-02
#       **: 1.00e-03 < p <= 1.00e-02
#      ***: 1.00e-04 < p <= 1.00e-03
#     ****: p <= 1.00e-04
formatted_star = []
for p in pvalues:
    if p > 0.05:
        formatted_star.append('ns')
    elif p > 0.01:
        formatted_star.append('*')
    elif p > 0.001:
        formatted_star.append('**')
    elif p > 0.0001:
        formatted_star.append('***')
    else:
        formatted_star.append('****')

# Plot with seaborn
ax = sns.boxplot(data=data, x=x, y=y, order=order, width=.5)

# Add annotations
annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
annotator.set_custom_annotations(formatted_star)
annotator.annotate()

# Label and show
# y-axis range from -0.25 to 0.3
ax.set_ylim(-0.2, 0.3)
# change the value of x-axis
plt.xticks(np.arange(3), ['SVM-linear', 'SVM-rbf', 'XGBoost'], rotation=45)
# name x-axis
plt.xlabel('Machine learning models')
# name y-axis
plt.ylabel('Cross-validated average $R^2$')
plt.tight_layout()
# plt.savefig("./plot1A.png", bbox_inches='tight')
plt.show()
plt.close()

# %%
df_Memory_1 = pd.concat([df_Memory_SVM_linear, df_Memory_SVM_rbf, df_Memory_XGBoost])  # , df_Memory_CV_AutoGluon

data = df_Memory_1
x = 'model'
y = 'test_r2'
order = ['SVM-linear', 'SVM-rbf', 'XGBoost']

pairs = [('SVM-linear', 'SVM-rbf'),
         ('SVM-rbf', 'XGBoost')]

pvalues = [5.683002e-04, 3.843204e-01]

formatted_pvalues = [f"p={p:.2e}" for p in pvalues]

formatted_star = []
for p in pvalues:
    if p > 0.05:
        formatted_star.append('ns')
    elif p > 0.01:
        formatted_star.append('*')
    elif p > 0.001:
        formatted_star.append('**')
    elif p > 0.0001:
        formatted_star.append('***')
    else:
        formatted_star.append('****')

# Plot with seaborn
ax = sns.boxplot(data=data, x=x, y=y, order=order, width=.5)

# Add annotations
annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
annotator.set_custom_annotations(formatted_star)
annotator.annotate()

# Label and show
ax.set_ylim(-0.2, 0.3)
# change the value of x-axis
plt.xticks(np.arange(3), ['SVM-linear', 'SVM-rbf', 'XGBoost'], rotation=45)
# name x-axis
plt.xlabel('Machine learning models')
# name y-axis
plt.ylabel('Cross-validated average $R^2$')
plt.tight_layout()
plt.show()
plt.close()

# %%
data = df_Stroop_1
x = 'model'
y = 'test_r_corr'
order = ['SVM-linear', 'SVM-rbf', 'XGBoost']

pairs = [('SVM-linear', 'SVM-rbf'),
         ('SVM-rbf', 'XGBoost')]

pvalues = [2.526188e-02, 5.154746e-01]

formatted_pvalues = [f"p={p:.2e}" for p in pvalues]

# creat formatted_star based on pvalues: p-value annotation legend:
#       ns: 5.00e-02 < p <= 1.00e+00
#        *: 1.00e-02 < p <= 5.00e-02
#       **: 1.00e-03 < p <= 1.00e-02
#      ***: 1.00e-04 < p <= 1.00e-03
#     ****: p <= 1.00e-04
formatted_star = []
for p in pvalues:
    if p > 0.05:
        formatted_star.append('ns')
    elif p > 0.01:
        formatted_star.append('*')
    elif p > 0.001:
        formatted_star.append('**')
    elif p > 0.0001:
        formatted_star.append('***')
    else:
        formatted_star.append('****')

# Plot with seaborn
ax = sns.boxplot(data=data, x=x, y=y, order=order, width=.5)

# Add annotations
annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
annotator.set_custom_annotations(formatted_star)
annotator.annotate()

# Label and show
# y-axis range from -0.25 to 0.3
ax.set_ylim(0, 0.6)
# change the value of x-axis
plt.xticks(np.arange(3), ['SVM-linear', 'SVM-rbf', 'XGBoost'], rotation=45)
# name x-axis
plt.xlabel('Machine learning models')
# name y-axis
plt.ylabel('Cross-validated average Pearson correlation coefficient')
plt.tight_layout()
# plt.savefig("./plot1A.png", bbox_inches='tight')
plt.show()
plt.close()

# %%
data = df_Memory_1
x = 'model'
y = 'test_r_corr'
order = ['SVM-linear', 'SVM-rbf', 'XGBoost']

pairs = [('SVM-linear', 'SVM-rbf'),
         ('SVM-rbf', 'XGBoost')]

pvalues = [2.526188e-02, 5.154746e-01]

formatted_pvalues = [f"p={p:.2e}" for p in pvalues]

formatted_star = []
for p in pvalues:
    if p > 0.05:
        formatted_star.append('ns')
    elif p > 0.01:
        formatted_star.append('*')
    elif p > 0.001:
        formatted_star.append('**')
    elif p > 0.0001:
        formatted_star.append('***')
    else:
        formatted_star.append('****')

# Plot with seaborn
ax = sns.boxplot(data=data, x=x, y=y, order=order, width=.5)

# Add annotations
annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
annotator.set_custom_annotations(formatted_star)
annotator.annotate()

# Label and show
# y-axis range from -0.25 to 0.3
ax.set_ylim(0, 0.6)
# change the value of x-axis
plt.xticks(np.arange(3), ['SVM-linear', 'SVM-rbf', 'XGBoost'], rotation=45)
# name x-axis
plt.xlabel('Machine learning models')
# name y-axis
plt.ylabel('Cross-validated average Pearson correlation coefficient')
plt.tight_layout()
# plt.savefig("./plot1A.png", bbox_inches='tight')
plt.show()
plt.close()
