import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')
import utils

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import julearn
import itertools
import starbars

from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
)
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# %%
main_results_feature_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov', 'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle']

# %%
model_list = ['XGBoost', 'AutoGluon']

print('\nStroop')
for model in model_list:
    for feature in main_results_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Stroop', 'SHIP_Trend', feature, 'scores.csv')
        if not os.path.exists(path):
            print(model, feature, 'missing')

print('\nMemory')
for model in model_list:
    for feature in main_results_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Memory', 'SHIP_Trend', feature, 'scores.csv')
        if not os.path.exists(path):
            print(model, feature, 'missing')

# %%
model = 'XGBoost'
df_Stroop_XGBoost_Sleep = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend_final/Sleep/scores.csv',
    index_col=0)
df_Stroop_XGBoost_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend_final/Cov/scores.csv',
    index_col=0)
df_Stroop_XGBoost_Sleep_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend_final/Sleep_Cov/scores.csv',
    index_col=0)
df_Stroop_XGBoost_Sleep_Shuffle_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend_final/Sleep_Shuffle_Cov/scores.csv',
    index_col=0)
df_Stroop_XGBoost_Brain = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend_final/Cov_Brain/scores.csv',
    index_col=0)
df_Stroop_XGBoost_Sleep_Cov_Brain = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend_final/Sleep_Cov_Brain/scores.csv',
    index_col=0)
df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend_final/Sleep_Cov_Brain_Shuffle/scores.csv',
    index_col=0)


df_Memory_XGBoost_Sleep = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Memory/SHIP_Trend_final/Sleep/scores.csv',
    index_col=0)
df_Memory_XGBoost_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Memory/SHIP_Trend_final/Cov/scores.csv',
    index_col=0)
df_Memory_XGBoost_Sleep_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Memory/SHIP_Trend_final/Sleep_Cov/scores.csv',
    index_col=0)
df_Memory_XGBoost_Sleep_Shuffle_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Memory/SHIP_Trend_final/Sleep_Shuffle_Cov/scores.csv',
    index_col=0)
df_Memory_XGBoost_Brain = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Memory/SHIP_Trend_final/Cov_Brain/scores.csv',
    index_col=0)
df_Memory_XGBoost_Sleep_Cov_Brain = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Memory/SHIP_Trend_final/Sleep_Cov_Brain/scores.csv',
    index_col=0)
df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Memory/SHIP_Trend_final/Sleep_Cov_Brain_Shuffle/scores.csv',
    index_col=0)

# %%
# for all dataframe, sort the rows by 'repeat' and 'fold' columns.
df_Stroop_XGBoost_Sleep = df_Stroop_XGBoost_Sleep.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_XGBoost_Cov = df_Stroop_XGBoost_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_XGBoost_Sleep_Cov = df_Stroop_XGBoost_Sleep_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_XGBoost_Sleep_Shuffle_Cov = df_Stroop_XGBoost_Sleep_Shuffle_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_XGBoost_Brain = df_Stroop_XGBoost_Brain.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_XGBoost_Sleep_Cov_Brain = df_Stroop_XGBoost_Sleep_Cov_Brain.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle = df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle.sort_values(by=['repeat', 'fold']).reset_index(drop=True)

df_Memory_XGBoost_Sleep = df_Memory_XGBoost_Sleep.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_XGBoost_Cov = df_Memory_XGBoost_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_XGBoost_Sleep_Cov = df_Memory_XGBoost_Sleep_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_XGBoost_Sleep_Shuffle_Cov = df_Memory_XGBoost_Sleep_Shuffle_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_XGBoost_Brain = df_Memory_XGBoost_Brain.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_XGBoost_Sleep_Cov_Brain = df_Memory_XGBoost_Sleep_Cov_Brain.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle = df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle.sort_values(by=['repeat', 'fold']).reset_index(drop=True)

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
# add 'n_train' and 'n_test' columns to every dataframe. The value is the number of training and testing samples
df_Stroop_XGBoost_Sleep['n_train'] = n_train
df_Stroop_XGBoost_Sleep['n_test'] = n_test
df_Stroop_XGBoost_Cov['n_train'] = n_train
df_Stroop_XGBoost_Cov['n_test'] = n_test
df_Stroop_XGBoost_Sleep_Cov['n_train'] = n_train
df_Stroop_XGBoost_Sleep_Cov['n_test'] = n_test
df_Stroop_XGBoost_Sleep_Shuffle_Cov['n_train'] = n_train
df_Stroop_XGBoost_Sleep_Shuffle_Cov['n_test'] = n_test
df_Stroop_XGBoost_Brain['n_train'] = n_train
df_Stroop_XGBoost_Brain['n_test'] = n_test
df_Stroop_XGBoost_Sleep_Cov_Brain['n_train'] = n_train
df_Stroop_XGBoost_Sleep_Cov_Brain['n_test'] = n_test
df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle['n_train'] = n_train
df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle['n_test'] = n_test

df_Memory_XGBoost_Sleep['n_train'] = n_train
df_Memory_XGBoost_Sleep['n_test'] = n_test
df_Memory_XGBoost_Cov['n_train'] = n_train
df_Memory_XGBoost_Cov['n_test'] = n_test
df_Memory_XGBoost_Sleep_Cov['n_train'] = n_train
df_Memory_XGBoost_Sleep_Cov['n_test'] = n_test
df_Memory_XGBoost_Sleep_Shuffle_Cov['n_train'] = n_train
df_Memory_XGBoost_Sleep_Shuffle_Cov['n_test'] = n_test
df_Memory_XGBoost_Brain['n_train'] = n_train
df_Memory_XGBoost_Brain['n_test'] = n_test
df_Memory_XGBoost_Sleep_Cov_Brain['n_train'] = n_train
df_Memory_XGBoost_Sleep_Cov_Brain['n_test'] = n_test
df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle['n_train'] = n_train
df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle['n_test'] = n_test

# %%
# add 'model' column to every dataframe. The value is each model's input feature
df_Stroop_XGBoost_Sleep['model'] = 'Sleep'
df_Stroop_XGBoost_Cov['model'] = 'Demographic (Demo)'
df_Stroop_XGBoost_Sleep_Cov['model'] = 'Sleep Demo'
df_Stroop_XGBoost_Sleep_Shuffle_Cov['model'] = 'Sleep (Shuffle) Demo'
df_Stroop_XGBoost_Brain['model'] = 'Brain'
df_Stroop_XGBoost_Sleep_Cov_Brain['model'] = 'Sleep Demo Brain'
df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle['model'] = 'Sleep Demo Brain (Shuffle)'

df_Memory_XGBoost_Sleep['model'] = 'Sleep'
df_Memory_XGBoost_Cov['model'] = 'Demographic (Demo)'
df_Memory_XGBoost_Sleep_Cov['model'] = 'Sleep Demo'
df_Memory_XGBoost_Sleep_Shuffle_Cov['model'] = 'Sleep (Shuffle) Demo'
df_Memory_XGBoost_Brain['model'] = 'Brain'
df_Memory_XGBoost_Sleep_Cov_Brain['model'] = 'Sleep Demo Brain'
df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle['model'] = 'Sleep Demo Brain (Shuffle)'

# %%
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
# add 'cv_mdsum' column to every dataframe. The value is cv_mdsum
df_Stroop_XGBoost_Sleep['cv_mdsum'] = cv_mdsum
df_Stroop_XGBoost_Cov['cv_mdsum'] = cv_mdsum
df_Stroop_XGBoost_Sleep_Cov['cv_mdsum'] = cv_mdsum
df_Stroop_XGBoost_Sleep_Shuffle_Cov['cv_mdsum'] = cv_mdsum
df_Stroop_XGBoost_Brain['cv_mdsum'] = cv_mdsum
df_Stroop_XGBoost_Sleep_Cov_Brain['cv_mdsum'] = cv_mdsum
df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle['cv_mdsum'] = cv_mdsum

df_Memory_XGBoost_Sleep['cv_mdsum'] = cv_mdsum
df_Memory_XGBoost_Cov['cv_mdsum'] = cv_mdsum
df_Memory_XGBoost_Sleep_Cov['cv_mdsum'] = cv_mdsum
df_Memory_XGBoost_Sleep_Shuffle_Cov['cv_mdsum'] = cv_mdsum
df_Memory_XGBoost_Brain['cv_mdsum'] = cv_mdsum
df_Memory_XGBoost_Sleep_Cov_Brain['cv_mdsum'] = cv_mdsum
df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle['cv_mdsum'] = cv_mdsum

# %%
from julearn.stats.corrected_ttest import corrected_ttest

# input all dataframes
stats_stroop_df = corrected_ttest(df_Stroop_XGBoost_Sleep, df_Stroop_XGBoost_Cov, df_Stroop_XGBoost_Sleep_Cov,
                                  df_Stroop_XGBoost_Sleep_Shuffle_Cov, df_Stroop_XGBoost_Brain,
                                  df_Stroop_XGBoost_Sleep_Cov_Brain, df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle)

stats_memory_df = corrected_ttest(df_Memory_XGBoost_Sleep, df_Memory_XGBoost_Cov, df_Memory_XGBoost_Sleep_Cov,
                                  df_Memory_XGBoost_Sleep_Shuffle_Cov, df_Memory_XGBoost_Brain,
                                  df_Memory_XGBoost_Sleep_Cov_Brain, df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle)

# %%
# add column 'significance' to the stats_stroop_df and stats_memory_df. If p-value larger than 0.05, the value is 'ns';
# if p-value larger than 0.01, the value is '*'; if p-value larger than 0.001, the value is '**'; if p-value larger than
# 0.0001, the value is '***'; if p-value smaller than 0.0001, the value is '****'.
stats_stroop_df['significance'] = stats_stroop_df['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')
stats_memory_df['significance'] = stats_memory_df['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')

# %%
# only keep rows which 'metric' is 'test_r_corr'
stats_stroop_df_cld = stats_stroop_df[stats_stroop_df['metric'] == 'test_r_corr']
stats_memory_df_cld = stats_memory_df[stats_memory_df['metric'] == 'test_r_corr']

# %%
"""
Make boxplot for Stroop and Memory
"""
df_Stroop = pd.concat([df_Stroop_XGBoost_Sleep, df_Stroop_XGBoost_Cov, df_Stroop_XGBoost_Sleep_Cov,
                       df_Stroop_XGBoost_Sleep_Shuffle_Cov, df_Stroop_XGBoost_Brain, df_Stroop_XGBoost_Sleep_Cov_Brain,
                       df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle])
df_Memory = pd.concat([df_Memory_XGBoost_Sleep, df_Memory_XGBoost_Cov, df_Memory_XGBoost_Sleep_Cov,
                       df_Memory_XGBoost_Sleep_Shuffle_Cov, df_Memory_XGBoost_Brain, df_Memory_XGBoost_Sleep_Cov_Brain,
                       df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle])

df_Stroop_Sleep_Compare = pd.concat([df_Stroop_XGBoost_Sleep, df_Stroop_XGBoost_Cov, df_Stroop_XGBoost_Sleep_Cov,
                                     df_Stroop_XGBoost_Sleep_Shuffle_Cov])
df_Memory_Sleep_Compare = pd.concat([df_Memory_XGBoost_Sleep, df_Memory_XGBoost_Cov, df_Memory_XGBoost_Sleep_Cov,
                                     df_Memory_XGBoost_Sleep_Shuffle_Cov])

df_Stroop_Brain_Compare = pd.concat([df_Stroop_XGBoost_Brain, df_Stroop_XGBoost_Sleep_Cov_Brain,
                                     df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle])
df_Memory_Brain_Compare = pd.concat([df_Memory_XGBoost_Brain, df_Memory_XGBoost_Sleep_Cov_Brain,
                                     df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle])

# %%
"""
Compare all main results feature together
"""
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

# Order of models to display on x-axis
order = ['Sleep', 'Demographic (Demo)', 'Sleep Demo', 'Sleep (Shuffle) Demo', 'Brain', 'Sleep Demo Brain',
         'Sleep Demo Brain (Shuffle)']

# Create a figure with two subplots in a row (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)  # sharey=True ensures same y-axis scale

# Boxplot for Stroop (no color, only outlines)
sns.boxplot(data=df_Stroop, x='model', y='test_r_corr', order=order, ax=axes[0], width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

# Swarmplot for Stroop with all points colored #FFD92F
sns.swarmplot(data=df_Stroop, x='model', y='test_r_corr', order=order, ax=axes[0],
              color="#fe9520", size=4)

axes[0].set_title('Stroop', fontsize=18)  # Set title font size
# axes[0].set_title('Stroop')  # Set title font size
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=18)  # Set y-axis label font size
# axes[0].set_ylabel('Pearson Correlation Coefficient')

# Boxplot for Memory (no color, only outlines)
sns.boxplot(data=df_Memory, x='model', y='test_r_corr', order=order, ax=axes[1], width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

# Swarmplot for Memory with all points colored #FFD92F
sns.swarmplot(data=df_Memory, x='model', y='test_r_corr', order=order, ax=axes[1],
              color="#fe9520", size=4)

axes[1].set_title('Memory', fontsize=18)  # Set title font size
# axes[1].set_title('Memory')  # Set title font size
axes[1].set_xlabel('')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.10, 0.60)  # Adjust the range according to your data

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=60, labelsize=14)  # Set x-axis tick label size
    # ax.tick_params(axis='x', rotation=45)  # Set x-axis tick label size
    ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'XGBoost_Stroop_Memory_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')

# Show the plot
plt.show()
plt.close()

# %%
"""
Plot for Stroop and Memory separately
"""
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

# Order of models to display on x-axis
order = ['Sleep', 'Demographic (Demo)', 'Sleep Demo', 'Sleep (Shuffle) Demo', 'Brain', 'Sleep Demo Brain',
         'Sleep Demo Brain (Shuffle)']

# Stroop plot only
fig, ax = plt.subplots(figsize=(12, 8))  # Create a single plot for Stroop

# Boxplot for Stroop (no color, only outlines)
sns.boxplot(data=df_Stroop, x='model', y='test_r_corr', order=order, ax=ax, width=0.3,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

# Swarmplot for Stroop with all points colored #FFD92F
sns.swarmplot(data=df_Stroop, x='model', y='test_r_corr', order=order, ax=ax,
              color="#FFD92F", size=4)

ax.set_title('Stroop', fontsize=18)  # Set title font size
ax.set_xlabel('')
ax.set_ylabel('Pearson Correlation Coefficient', fontsize=18)  # Set y-axis label font size

# Rotate x-axis labels for better readability
ax.tick_params(axis='x', rotation=45, labelsize=16)  # Set x-axis tick label size
ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Set y-axis scale
ax.set_ylim(-0.10, 0.60)  # Adjust the range according to your data

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'XGBoost_Stroop_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')

# Show the plot
plt.show()
plt.close()

# Memory plot only
fig, ax = plt.subplots(figsize=(12, 8))  # Create a single plot for Memory

# Boxplot for Memory (no color, only outlines)
sns.boxplot(data=df_Memory, x='model', y='test_r_corr', order=order, ax=ax, width=0.3,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

# Swarmplot for Memory with all points colored #FFD92F
sns.swarmplot(data=df_Memory, x='model', y='test_r_corr', order=order, ax=ax,
              color="#FFD92F", size=4)

ax.set_title('Memory', fontsize=18)  # Set title font size
ax.set_xlabel('')
ax.set_ylabel('Pearson Correlation Coefficient', fontsize=18)  # Set y-axis label font size

# Rotate x-axis labels for better readability
ax.tick_params(axis='x', rotation=45, labelsize=16)  # Set x-axis tick label size
ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Set y-axis scale
ax.set_ylim(-0.10, 0.60)  # Adjust the range according to your data

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
plt.savefig(stats_fig_path + 'XGBoost_Memory_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')

# Show the plot
plt.show()
plt.close()

# %%
"""
Stroop, Sleep and Brain comparison separately
"""
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

# Order of models to display on x-axis
order_1 = ['Sleep', 'Demographic (Demo)', 'Sleep Demo', 'Sleep (Shuffle) Demo']
order_2 = ['Brain', 'Sleep Demo Brain',
         'Sleep Demo Brain (Shuffle)']

# Create a figure with two subplots in a row (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)  # sharey=True ensures same y-axis scale

# Boxplot for Stroop (no color, only outlines)
sns.boxplot(data=df_Stroop_Sleep_Compare, x='model', y='test_r_corr', order=order_1, ax=axes[0], width=0.4,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

# Swarmplot for Stroop with all points colored #FFD92F
sns.swarmplot(data=df_Stroop_Sleep_Compare, x='model', y='test_r_corr', order=order_1, ax=axes[0],
              color="#FFD92F", size=4)

axes[0].set_title('Stroop', fontsize=18)  # Set title font size
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=18)  # Set y-axis label font size

# Boxplot for Memory (no color, only outlines)
sns.boxplot(data=df_Stroop_Brain_Compare, x='model', y='test_r_corr', order=order_2, ax=axes[1], width=0.3,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

# Swarmplot for Memory with all points colored #FFD92F
sns.swarmplot(data=df_Stroop_Brain_Compare, x='model', y='test_r_corr', order=order_2, ax=axes[1],
              color="#FFD92F", size=4)

axes[1].set_title('Stroop', fontsize=18)  # Set title font size
axes[1].set_xlabel('')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.10, 0.60)  # Adjust the range according to your data

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=45, labelsize=16)  # Set x-axis tick label size
    ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'Stroop_Sleep_Brain_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')

# Show the plot
plt.show()
plt.close()

# %%
"""
Memory, Sleep and Brain comparison separately
"""
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

# Order of models to display on x-axis
order_1 = ['Sleep', 'Demographic (Demo)', 'Sleep Demo', 'Sleep (Shuffle) Demo']
order_2 = ['Brain', 'Sleep Demo Brain',
         'Sleep Demo Brain (Shuffle)']

# Create a figure with two subplots in a row (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)  # sharey=True ensures same y-axis scale

# Boxplot for Memory (no color, only outlines)
sns.boxplot(data=df_Memory_Sleep_Compare, x='model', y='test_r_corr', order=order_1, ax=axes[0], width=0.4,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

# Swarmplot for Memory with all points colored #FFD92F
sns.swarmplot(data=df_Memory_Sleep_Compare, x='model', y='test_r_corr', order=order_1, ax=axes[0],
              color="#FFD92F", size=4)

axes[0].set_title('Memory', fontsize=18)  # Set title font size
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=18)  # Set y-axis label font size

# Boxplot for Memory (no color, only outlines)
sns.boxplot(data=df_Memory_Brain_Compare, x='model', y='test_r_corr', order=order_2, ax=axes[1], width=0.3,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))

# Swarmplot for Memory with all points colored #FFD92F
sns.swarmplot(data=df_Memory_Brain_Compare, x='model', y='test_r_corr', order=order_2, ax=axes[1],
              color="#FFD92F", size=4)

axes[1].set_title('Memory', fontsize=18)  # Set title font size
axes[1].set_xlabel('')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.10, 0.60)  # Adjust the range according to your data

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=45, labelsize=16)  # Set x-axis tick label size
    ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'Memory_Sleep_Brain_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')

# Show the plot
plt.show()
plt.close()