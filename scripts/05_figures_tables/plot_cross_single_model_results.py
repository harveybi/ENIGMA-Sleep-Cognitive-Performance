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
model_list = ['Dummy', 'Linear', 'Ridge', 'SVM-linear', 'SVM-rbf', 'rf', 'XGBoost', 'AutoGluon']

# %%
"""
Compare the performance of different models, feature 'Sleep_Cov_Brain', target 'Stroop', then 'Memory'
Compare the performance of different feature combinations,  AutoGluon first, then XGBoost
Compare based on test_r_corr first, then test_r2
"""
# load results of different models for feature 'Sleep_Cov_Brain', target 'Stroop' and 'Memory'
feature = 'Sleep_Cov_Brain'
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

cv_mdsum = df_Stroop_Ridge['cv_mdsum'][0]

df_Stroop_XGBoost['cv_mdsum'] = cv_mdsum
df_Memory_XGBoost['cv_mdsum'] = cv_mdsum

df_Stroop_CV_AutoGluon['cv_mdsum'] = cv_mdsum
df_Memory_CV_AutoGluon['cv_mdsum'] = cv_mdsum

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

stats_stroop_df_1['significance'] = stats_stroop_df_1['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')
stats_memory_df_1['significance'] = stats_memory_df_1['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')

# for the metric column in stats_stroop_df, only keep the rows which values are test_r_corr and test_r2.
stats_stroop_df_1_cld = stats_stroop_df_1[stats_stroop_df_1['metric'].isin(['test_r_corr', 'test_r2'])]
# add a new column 'significant' to stats_stroop_df_cld. If 'p-val-corrected' < 0.05, 'significant' is True, otherwise False.
stats_stroop_df_1_cld['significant'] = stats_stroop_df_1_cld['p-val-corrected'] < 0.05

stats_memory_df_1_cld = stats_memory_df_1[stats_memory_df_1['metric'].isin(['test_r_corr', 'test_r2'])]
# add a new column 'significant' to stats_memory_df_cld. If 'p-val-corrected' < 0.05, 'significant' is True, otherwise False.
stats_memory_df_1_cld['significant'] = stats_memory_df_1_cld['p-val-corrected'] < 0.05

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

stats_stroop_df_2 = corrected_ttest(df_Stroop_Linear, df_Stroop_Ridge, df_Stroop_SVM_linear, df_Stroop_SVM_rbf,
                                    df_Stroop_rf, df_Stroop_XGBoost, df_Stroop_CV_AutoGluon)
stats_memory_df_2 = corrected_ttest(df_Memory_Linear, df_Memory_Ridge, df_Memory_SVM_linear, df_Memory_SVM_rbf,
                                    df_Memory_rf, df_Memory_XGBoost, df_Memory_CV_AutoGluon)

stats_stroop_df_2['significance'] = stats_stroop_df_2['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')
stats_memory_df_2['significance'] = stats_memory_df_2['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')

stats_stroop_df_2_cld = stats_stroop_df_2[stats_stroop_df_2['metric'].isin(['test_r_corr', 'test_r2'])]
stats_stroop_df_2_cld['significant'] = stats_stroop_df_2_cld['p-val-corrected'] < 0.05

stats_memory_df_2_cld = stats_memory_df_2[stats_memory_df_2['metric'].isin(['test_r_corr', 'test_r2'])]
stats_memory_df_2_cld['significant'] = stats_memory_df_2_cld['p-val-corrected'] < 0.05

# %%
df_Stroop_Sleep_Cov_Brain_model_compare = pd.concat([
    df_Stroop_Linear, df_Stroop_Ridge, df_Stroop_SVM_linear, df_Stroop_SVM_rbf, df_Stroop_rf, df_Stroop_XGBoost,
    df_Stroop_CV_AutoGluon
])
df_Memory_Sleep_Cov_Brain_model_compare = pd.concat([
    df_Memory_Linear, df_Memory_Ridge, df_Memory_SVM_linear, df_Memory_SVM_rbf, df_Memory_rf, df_Memory_XGBoost,
    df_Memory_CV_AutoGluon
])
# --- above are dataframe for stats and plots for model comparison for feature 'Sleep_Cov_Brain' and target 'Stroop' and 'Memory' --- #

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
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend_final/Brain/scores.csv',
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
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Memory/SHIP_Trend_final/Brain/scores.csv',
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
df_Stroop_XGBoost_Sleep['model'] = 'Sleep'
df_Stroop_XGBoost_Cov['model'] = 'Demographic'
df_Stroop_XGBoost_Sleep_Cov['model'] = 'Sleep, Demo'
df_Stroop_XGBoost_Sleep_Shuffle_Cov['model'] = 'Sleep (Shuffle), Demo'
df_Stroop_XGBoost_Brain['model'] = 'Brain'
df_Stroop_XGBoost_Sleep_Cov_Brain['model'] = 'Sleep, Demo, Brain'
df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle['model'] = 'Sleep, Demo, Brain (Shuffle)'

df_Memory_XGBoost_Sleep['model'] = 'Sleep'
df_Memory_XGBoost_Cov['model'] = 'Demographic'
df_Memory_XGBoost_Sleep_Cov['model'] = 'Sleep, Demo'
df_Memory_XGBoost_Sleep_Shuffle_Cov['model'] = 'Sleep (Shuffle), Demo'
df_Memory_XGBoost_Brain['model'] = 'Brain'
df_Memory_XGBoost_Sleep_Cov_Brain['model'] = 'Sleep, Demo, Brain'
df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle['model'] = 'Sleep, Demo, Brain (Shuffle)'

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
stats_stroop_df = corrected_ttest(df_Stroop_XGBoost_Sleep, df_Stroop_XGBoost_Cov, df_Stroop_XGBoost_Sleep_Cov,
                                  df_Stroop_XGBoost_Sleep_Shuffle_Cov, df_Stroop_XGBoost_Brain,
                                  df_Stroop_XGBoost_Sleep_Cov_Brain, df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle)

stats_memory_df = corrected_ttest(df_Memory_XGBoost_Sleep, df_Memory_XGBoost_Cov, df_Memory_XGBoost_Sleep_Cov,
                                  df_Memory_XGBoost_Sleep_Shuffle_Cov, df_Memory_XGBoost_Brain,
                                  df_Memory_XGBoost_Sleep_Cov_Brain, df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle)

stats_stroop_df['significance'] = stats_stroop_df['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')
stats_memory_df['significance'] = stats_memory_df['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')

stats_stroop_df_cld = stats_stroop_df[stats_stroop_df['metric'] == 'test_r_corr']
stats_memory_df_cld = stats_memory_df[stats_memory_df['metric'] == 'test_r_corr']

# %%
df_Stroop_XGBoost_feature_compare = pd.concat([
    df_Stroop_XGBoost_Sleep, df_Stroop_XGBoost_Cov, df_Stroop_XGBoost_Sleep_Cov, df_Stroop_XGBoost_Sleep_Shuffle_Cov,
    df_Stroop_XGBoost_Brain, df_Stroop_XGBoost_Sleep_Cov_Brain, df_Stroop_XGBoost_Sleep_Cov_Brain_Shuffle
])
df_Memory_XGBoost_feature_compare = pd.concat([
    df_Memory_XGBoost_Sleep, df_Memory_XGBoost_Cov, df_Memory_XGBoost_Sleep_Cov, df_Memory_XGBoost_Sleep_Shuffle_Cov,
    df_Memory_XGBoost_Brain, df_Memory_XGBoost_Sleep_Cov_Brain, df_Memory_XGBoost_Sleep_Cov_Brain_Shuffle
])
# --- above are dataframe for stats and plots for feature comparison for model 'XGBoost' and target 'Stroop' and 'Memory' --- #

# %%
"""
Make plots.
Figure 1: A. Stroop prediction, model comparison; B. Stroop prediction, feature comparison.
Figure 2: A. Memory prediction, model comparison; B. Memory prediction, feature comparison.
"""
# Figure 1
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

custom_colors = {
    'Linear Regression': '#b7d7e0',
    'Ridge Regression': '#8ec4d5',
    'SVM-linear': '#82c2f3',
    'SVM-rbf': '#ffd7bb',
    'Random Forest': '#fea28f',
    'XGBoost': '#fd7351',
    'AutoGluon': '#fd2723'
}

# Order of models to display on x-axis
order_feature = ['Sleep', 'Demographic', 'Sleep, Demo', 'Sleep (Shuffle), Demo', 'Brain', 'Sleep, Demo, Brain',
                 'Sleep, Demo, Brain (Shuffle)']
order_model = ['Linear Regression', 'Ridge Regression', 'SVM-linear', 'SVM-rbf', 'Random Forest', 'XGBoost',
               'AutoGluon']

fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True)

# axes[0] for model comparison
sns.boxplot(data=df_Stroop_Sleep_Cov_Brain_model_compare, x='model', y='test_r_corr', order=order_model, ax=axes[0],
            width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
swarmplot = sns.swarmplot(data=df_Stroop_Sleep_Cov_Brain_model_compare, x='model', y='test_r_corr', order=order_model,
                          ax=axes[0], size=4, palette=custom_colors)  # Assign colors directly

# axes[0].set_title('Stroop', fontsize=18)  # Set title font size
# axes[0].set_xlabel('Model', fontsize=14)  # Set x-axis label font size
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=18)

# axes[1] for feature comparison
sns.boxplot(data=df_Stroop_XGBoost_feature_compare, x='model', y='test_r_corr', order=order_feature, ax=axes[1],
            width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
sns.swarmplot(data=df_Stroop_XGBoost_feature_compare, x='model', y='test_r_corr', order=order_feature, ax=axes[1],
              color="#fd7351", size=4)
# axes[1].set_title('Memory', fontsize=18)  # Set title font size
# axes[1].set_title('Memory')  # Set title font size
axes[1].set_xlabel('')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.10, 0.55)

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=60, labelsize=14)  # Set x-axis tick label size
    # ax.tick_params(axis='x', rotation=45)  # Set x-axis tick label size
    ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'Stroop_XGBoost_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')

plt.show()
plt.close()

# %%
# Figure 2
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

# Order of models to display on x-axis
order_feature = ['Sleep', 'Demographic', 'Sleep, Demo', 'Sleep (Shuffle), Demo', 'Brain', 'Sleep, Demo, Brain',
                 'Sleep, Demo, Brain (Shuffle)']
order_model = ['Linear Regression', 'Ridge Regression', 'SVM-linear', 'SVM-rbf', 'Random Forest', 'XGBoost',
               'AutoGluon']

fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True)

# axes[0] for model comparison
sns.boxplot(data=df_Memory_Sleep_Cov_Brain_model_compare, x='model', y='test_r_corr', order=order_model, ax=axes[0],
            width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
sns.swarmplot(data=df_Memory_Sleep_Cov_Brain_model_compare, x='model', y='test_r_corr', order=order_model, ax=axes[0],
              palette=custom_colors, size=4)
# axes[0].set_title('Stroop', fontsize=18)  # Set title font size
# axes[0].set_xlabel('Model', fontsize=14)  # Set x-axis label font size
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=18)

# axes[1] for feature comparison
sns.boxplot(data=df_Memory_XGBoost_feature_compare, x='model', y='test_r_corr', order=order_feature, ax=axes[1],
            width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
sns.swarmplot(data=df_Memory_XGBoost_feature_compare, x='model', y='test_r_corr', order=order_feature, ax=axes[1],
              color="#fd7351", size=4)
# axes[1].set_title('Memory', fontsize=18)  # Set title font size
# axes[1].set_title('Memory')  # Set title font size
axes[1].set_xlabel('')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.15, 0.50)

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=60, labelsize=14)  # Set x-axis tick label size
    # ax.tick_params(axis='x', rotation=45)  # Set x-axis tick label size
    ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'Memory_XGBoost_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')

plt.show()
plt.close()

# %%
"""
TODO: model = 'AutoGluon'
"""
model = 'AutoGluon'

df_Stroop_AutoGluon_Sleep = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Stroop/SHIP_Trend/Sleep/scores.csv')
df_Stroop_AutoGluon_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Stroop/SHIP_Trend/Cov/scores.csv')
df_Stroop_AutoGluon_Sleep_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Stroop/SHIP_Trend/Sleep_Cov/scores.csv')
df_Stroop_AutoGluon_Sleep_Shuffle_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Stroop/SHIP_Trend/Sleep_Shuffle_Cov/scores.csv')
df_Stroop_AutoGluon_Brain = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Stroop/SHIP_Trend/Brain/scores.csv')
df_Stroop_AutoGluon_Sleep_Cov_Brain = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Stroop/SHIP_Trend/Sleep_Cov_Brain/scores.csv')
df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Stroop/SHIP_Trend/Sleep_Cov_Brain_Shuffle/scores.csv')

df_Memory_AutoGluon_Sleep = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Memory/SHIP_Trend/Sleep/scores.csv')
df_Memory_AutoGluon_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Memory/SHIP_Trend/Cov/scores.csv')
df_Memory_AutoGluon_Sleep_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Memory/SHIP_Trend/Sleep_Cov/scores.csv')
df_Memory_AutoGluon_Sleep_Shuffle_Cov = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Memory/SHIP_Trend/Sleep_Shuffle_Cov/scores.csv')
df_Memory_AutoGluon_Brain = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Memory/SHIP_Trend/Brain/scores.csv')
df_Memory_AutoGluon_Sleep_Cov_Brain = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Memory/SHIP_Trend/Sleep_Cov_Brain/scores.csv')
df_Memory_AutoGluon_Sleep_Cov_Brain_Shuffle = pd.read_csv(
    '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Memory/SHIP_Trend/Sleep_Cov_Brain_Shuffle/scores.csv')

# %%
# for all dataframe, sort the rows by 'repeat' and 'fold' columns.
df_Stroop_AutoGluon_Sleep = df_Stroop_AutoGluon_Sleep.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_AutoGluon_Cov = df_Stroop_AutoGluon_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_AutoGluon_Sleep_Cov = df_Stroop_AutoGluon_Sleep_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_AutoGluon_Sleep_Shuffle_Cov = df_Stroop_AutoGluon_Sleep_Shuffle_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_AutoGluon_Brain = df_Stroop_AutoGluon_Brain.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_AutoGluon_Sleep_Cov_Brain = df_Stroop_AutoGluon_Sleep_Cov_Brain.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle = df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle.sort_values(by=['repeat', 'fold']).reset_index(drop=True)

df_Memory_AutoGluon_Sleep = df_Memory_AutoGluon_Sleep.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_AutoGluon_Cov = df_Memory_AutoGluon_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_AutoGluon_Sleep_Cov = df_Memory_AutoGluon_Sleep_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_AutoGluon_Sleep_Shuffle_Cov = df_Memory_AutoGluon_Sleep_Shuffle_Cov.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_AutoGluon_Brain = df_Memory_AutoGluon_Brain.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Memory_AutoGluon_Sleep_Cov_Brain = df_Memory_AutoGluon_Sleep_Cov_Brain.sort_values(by=['repeat', 'fold']).reset_index(drop=True)
df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle = df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle.sort_values(by=['repeat', 'fold']).reset_index(drop=True)

# %%
df_Stroop_AutoGluon_Sleep['n_train'] = n_train
df_Stroop_AutoGluon_Sleep['n_test'] = n_test
df_Stroop_AutoGluon_Cov['n_train'] = n_train
df_Stroop_AutoGluon_Cov['n_test'] = n_test
df_Stroop_AutoGluon_Sleep_Cov['n_train'] = n_train
df_Stroop_AutoGluon_Sleep_Cov['n_test'] = n_test
df_Stroop_AutoGluon_Sleep_Shuffle_Cov['n_train'] = n_train
df_Stroop_AutoGluon_Sleep_Shuffle_Cov['n_test'] = n_test
df_Stroop_AutoGluon_Brain['n_train'] = n_train
df_Stroop_AutoGluon_Brain['n_test'] = n_test
df_Stroop_AutoGluon_Sleep_Cov_Brain['n_train'] = n_train
df_Stroop_AutoGluon_Sleep_Cov_Brain['n_test'] = n_test
df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle['n_train'] = n_train
df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle['n_test'] = n_test

df_Memory_AutoGluon_Sleep['n_train'] = n_train
df_Memory_AutoGluon_Sleep['n_test'] = n_test
df_Memory_AutoGluon_Cov['n_train'] = n_train
df_Memory_AutoGluon_Cov['n_test'] = n_test
df_Memory_AutoGluon_Sleep_Cov['n_train'] = n_train
df_Memory_AutoGluon_Sleep_Cov['n_test'] = n_test
df_Memory_AutoGluon_Sleep_Shuffle_Cov['n_train'] = n_train
df_Memory_AutoGluon_Sleep_Shuffle_Cov['n_test'] = n_test
df_Memory_AutoGluon_Brain['n_train'] = n_train
df_Memory_AutoGluon_Brain['n_test'] = n_test
df_Memory_AutoGluon_Sleep_Cov_Brain['n_train'] = n_train
df_Memory_AutoGluon_Sleep_Cov_Brain['n_test'] = n_test
df_Memory_AutoGluon_Sleep_Cov_Brain_Shuffle['n_train'] = n_train
df_Memory_AutoGluon_Sleep_Cov_Brain_Shuffle['n_test'] = n_test

# %%
df_Stroop_AutoGluon_Sleep['model'] = 'Sleep'
df_Stroop_AutoGluon_Cov['model'] = 'Demographic'
df_Stroop_AutoGluon_Sleep_Cov['model'] = 'Sleep, Demo'
df_Stroop_AutoGluon_Sleep_Shuffle_Cov['model'] = 'Sleep (Shuffle), Demo'
df_Stroop_AutoGluon_Brain['model'] = 'Brain'
df_Stroop_AutoGluon_Sleep_Cov_Brain['model'] = 'Sleep, Demo, Brain'
df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle['model'] = 'Sleep, Demo, Brain (Shuffle)'

df_Memory_AutoGluon_Sleep['model'] = 'Sleep'
df_Memory_AutoGluon_Cov['model'] = 'Demographic'
df_Memory_AutoGluon_Sleep_Cov['model'] = 'Sleep, Demo'
df_Memory_AutoGluon_Sleep_Shuffle_Cov['model'] = 'Sleep (Shuffle), Demo'
df_Memory_AutoGluon_Brain['model'] = 'Brain'
df_Memory_AutoGluon_Sleep_Cov_Brain['model'] = 'Sleep, Demo, Brain'
df_Memory_AutoGluon_Sleep_Cov_Brain_Shuffle['model'] = 'Sleep, Demo, Brain (Shuffle)'

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

df_Stroop_AutoGluon_Sleep['cv_mdsum'] = cv_mdsum
df_Stroop_AutoGluon_Cov['cv_mdsum'] = cv_mdsum
df_Stroop_AutoGluon_Sleep_Cov['cv_mdsum'] = cv_mdsum
df_Stroop_AutoGluon_Sleep_Shuffle_Cov['cv_mdsum'] = cv_mdsum
df_Stroop_AutoGluon_Brain['cv_mdsum'] = cv_mdsum
df_Stroop_AutoGluon_Sleep_Cov_Brain['cv_mdsum'] = cv_mdsum
df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle['cv_mdsum'] = cv_mdsum

df_Memory_AutoGluon_Sleep['cv_mdsum'] = cv_mdsum
df_Memory_AutoGluon_Cov['cv_mdsum'] = cv_mdsum
df_Memory_AutoGluon_Sleep_Cov['cv_mdsum'] = cv_mdsum
df_Memory_AutoGluon_Sleep_Shuffle_Cov['cv_mdsum'] = cv_mdsum
df_Memory_AutoGluon_Brain['cv_mdsum'] = cv_mdsum
df_Memory_AutoGluon_Sleep_Cov_Brain['cv_mdsum'] = cv_mdsum
df_Memory_AutoGluon_Sleep_Cov_Brain_Shuffle['cv_mdsum'] = cv_mdsum

# %%
stats_stroop_df = corrected_ttest(df_Stroop_AutoGluon_Sleep, df_Stroop_AutoGluon_Cov, df_Stroop_AutoGluon_Sleep_Cov,
                                  df_Stroop_AutoGluon_Sleep_Shuffle_Cov, df_Stroop_AutoGluon_Brain,
                                  df_Stroop_AutoGluon_Sleep_Cov_Brain, df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle)

stats_memory_df = corrected_ttest(df_Memory_AutoGluon_Sleep, df_Memory_AutoGluon_Cov, df_Memory_AutoGluon_Sleep_Cov,
                                  df_Memory_AutoGluon_Sleep_Shuffle_Cov, df_Memory_AutoGluon_Brain,
                                  df_Memory_AutoGluon_Sleep_Cov_Brain, df_Memory_AutoGluon_Sleep_Cov_Brain_Shuffle)

stats_stroop_df['significance'] = stats_stroop_df['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')
stats_memory_df['significance'] = stats_memory_df['p-val-corrected'].apply(lambda x: 'ns' if x > 0.05 else '*' if x > 0.01 else '**' if x > 0.001 else '***' if x > 0.0001 else '****')

stats_stroop_df_cld = stats_stroop_df[stats_stroop_df['metric'] == 'test_r_corr']
stats_memory_df_cld = stats_memory_df[stats_memory_df['metric'] == 'test_r_corr']

# %%
df_Stroop_AutoGluon_feature_compare = pd.concat([
    df_Stroop_AutoGluon_Sleep, df_Stroop_AutoGluon_Cov, df_Stroop_AutoGluon_Sleep_Cov, df_Stroop_AutoGluon_Sleep_Shuffle_Cov,
    df_Stroop_AutoGluon_Brain, df_Stroop_AutoGluon_Sleep_Cov_Brain, df_Stroop_AutoGluon_Sleep_Cov_Brain_Shuffle
])
df_Memory_AutoGluon_feature_compare = pd.concat([
    df_Memory_AutoGluon_Sleep, df_Memory_AutoGluon_Cov, df_Memory_AutoGluon_Sleep_Cov, df_Memory_AutoGluon_Sleep_Shuffle_Cov,
    df_Memory_AutoGluon_Brain, df_Memory_AutoGluon_Sleep_Cov_Brain, df_Memory_AutoGluon_Sleep_Cov_Brain_Shuffle
])
# --- above are dataframe for stats and plots for feature comparison for model 'AutoGluon' and target 'Stroop' and 'Memory' --- #

# %%
"""
Make plots.
Figure 1: A. Stroop prediction, model comparison; B. Stroop prediction, feature comparison.
Figure 2: A. Memory prediction, model comparison; B. Memory prediction, feature comparison.
"""
# Figure 1
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

custom_colors = {
    'Linear Regression': '#b7d7e0',
    'Ridge Regression': '#8ec4d5',
    'SVM-linear': '#82c2f3',
    'SVM-rbf': '#ffd7bb',
    'Random Forest': '#fea28f',
    'XGBoost': '#fd7351',
    'AutoGluon': '#fd2723'
}

# Order of models to display on x-axis
order_feature = ['Sleep', 'Demographic', 'Sleep, Demo', 'Sleep (Shuffle), Demo', 'Brain', 'Sleep, Demo, Brain',
                 'Sleep, Demo, Brain (Shuffle)']
order_model = ['Linear Regression', 'Ridge Regression', 'SVM-linear', 'SVM-rbf', 'Random Forest', 'XGBoost',
               'AutoGluon']

fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True)

# axes[0] for model comparison
sns.boxplot(data=df_Stroop_Sleep_Cov_Brain_model_compare, x='model', y='test_r_corr', order=order_model, ax=axes[0],
            width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
swarmplot = sns.swarmplot(data=df_Stroop_Sleep_Cov_Brain_model_compare, x='model', y='test_r_corr', order=order_model,
                          ax=axes[0], size=4, palette=custom_colors)  # Assign colors directly

# axes[0].set_title('Stroop', fontsize=18)  # Set title font size
# axes[0].set_xlabel('Model', fontsize=14)  # Set x-axis label font size
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=18)

# axes[1] for feature comparison
sns.boxplot(data=df_Stroop_AutoGluon_feature_compare, x='model', y='test_r_corr', order=order_feature, ax=axes[1],
            width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
sns.swarmplot(data=df_Stroop_AutoGluon_feature_compare, x='model', y='test_r_corr', order=order_feature, ax=axes[1],
              color="#fd7351", size=4)
# axes[1].set_title('Memory', fontsize=18)  # Set title font size
# axes[1].set_title('Memory')  # Set title font size
axes[1].set_xlabel('')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.10, 0.55)

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=60, labelsize=14)  # Set x-axis tick label size
    # ax.tick_params(axis='x', rotation=45)  # Set x-axis tick label size
    ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'Stroop_AutoGluon_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')

plt.show()
plt.close()

# %%
# Figure 2
# Set the seaborn theme to remove top and right spines
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Globally set font size for labels, titles, and ticks
sns.set_context("paper")  # 'talk' context is good for figures; font_scale adjusts overall size

# Order of models to display on x-axis
order_feature = ['Sleep', 'Demographic', 'Sleep, Demo', 'Sleep (Shuffle), Demo', 'Brain', 'Sleep, Demo, Brain',
                 'Sleep, Demo, Brain (Shuffle)']
order_model = ['Linear Regression', 'Ridge Regression', 'SVM-linear', 'SVM-rbf', 'Random Forest', 'XGBoost',
               'AutoGluon']

fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True)

# axes[0] for model comparison
sns.boxplot(data=df_Memory_Sleep_Cov_Brain_model_compare, x='model', y='test_r_corr', order=order_model, ax=axes[0],
            width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
sns.swarmplot(data=df_Memory_Sleep_Cov_Brain_model_compare, x='model', y='test_r_corr', order=order_model, ax=axes[0],
              palette=custom_colors, size=4)
# axes[0].set_title('Stroop', fontsize=18)  # Set title font size
# axes[0].set_xlabel('Model', fontsize=14)  # Set x-axis label font size
axes[0].set_xlabel('')
axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=18)

# axes[1] for feature comparison
sns.boxplot(data=df_Memory_AutoGluon_feature_compare, x='model', y='test_r_corr', order=order_feature, ax=axes[1],
            width=0.6,
            showfliers=False,  # Hides outliers in the boxplot since swarmplot shows them
            boxprops=dict(facecolor='none'),  # No fill color
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'))
sns.swarmplot(data=df_Memory_AutoGluon_feature_compare, x='model', y='test_r_corr', order=order_feature, ax=axes[1],
              color="#fd7351", size=4)
# axes[1].set_title('Memory', fontsize=18)  # Set title font size
# axes[1].set_title('Memory')  # Set title font size
axes[1].set_xlabel('')
axes[1].set_ylabel('')  # No y-label for the second subplot since they share y-axis

# Set y-axis scale to be the same across both subplots
axes[0].set_ylim(-0.15, 0.50)

# Rotate x-axis labels for better readability
for ax in axes:
    ax.tick_params(axis='x', rotation=60, labelsize=14)  # Set x-axis tick label size
    # ax.tick_params(axis='x', rotation=45)  # Set x-axis tick label size
    ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# Adjust layout to ensure no overlap
plt.tight_layout()

# Save as SVG
stats_fig_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/'
plt.savefig(stats_fig_path + 'Memory_AutoGluon_Pearson_corr.svg', format='svg', dpi=1200, bbox_inches='tight')

plt.show()
plt.close()
