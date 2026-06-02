import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import ml_pipeline
from ml_pipeline import MLPipeline

import argparse
import pandas as pd

# %%
# reload utils
import importlib
importlib.reload(utils)
importlib.reload(ml_pipeline)
import ml_pipeline
from ml_pipeline import MLPipeline

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='Ridge Regression for n-back working memory in Liege dataset.')
parser.add_argument('feature_comb', type=str, help='''Name of the feature combination to be used.
    Available combinations are:
    - Sleep: Uses sleep features only.
    - Cov: Uses covariates only.
    - Sleep_Cov: Uses sleep features combined with covariates.
    - Sleep_Cov_Brain: Uses sleep features, covariates, and brain imaging data.
    - Sleep_Cov_CT: Uses sleep features, covariates, and cortical thickness from imaging data.
    - Sleep_Cov_SA: Uses sleep features, covariates, and surface area from imaging data.
    - Sleep_Cov_Subcor: Uses sleep features, covariates, and subcortical volumes.
    - Cov_Brain: Uses covariates combined with brain imaging data.
    - Cov_CT: Uses covariates and cortical thickness from imaging data.
    - Cov_SA: Uses covariates and surface area from imaging data.
    - Cov_Subcor: Uses covariates and subcortical volumes.
    - Sleep_Shuffle_Cov: Uses sleep features (shuffled) and covariates.
    - Sleep_Cov_Subcor_Shuffle: Uses sleep features, covariates, and subcortical volumes (shuffled).
    - Cov_Brain_Shuffle: Uses covariates and brain imaging data (shuffled).
    - Cov_Subcor_Shuffle: Uses covariates and subcortical volumes (shuffled).
''')

parser.add_argument('target', type=str, help='''Name of the target to be predicted.
    Available targets are:
    - Stroop: Stroop_Test
    - Memory: Average n-back working memory accuracy
    - 1-back: n-back working memory accuracy for 1-back condition
    - 2-back: n-back working memory accuracy for 2-back condition
    - 3-back: n-back working memory accuracy for 3-back condition
''')

# Add num_cores as an optional argument with a default value
parser.add_argument('num_cores', type=int, default=1,
                    help='Number of cores to be used for parallel processing. Default is 1.')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target
num_cores = args.num_cores

print(f"\nStarting Ridge regression pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/Ridge/{target}/Liege/'

# %%
# df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed.csv')
df_Liege_cleaned = pd.read_csv(data_save_path + 'Liege_dataset_renamed_cleaned.csv')

# %%
# column '1-back_acc', '2-back_acc', '3-back_acc' * 100
df_Liege_cleaned['1-back_acc'] = df_Liege_cleaned['1-back_acc'] * 100
df_Liege_cleaned['2-back_acc'] = df_Liege_cleaned['2-back_acc'] * 100
df_Liege_cleaned['3-back_acc'] = df_Liege_cleaned['3-back_acc'] * 100

# %%
# add a column of 'Age_Group' after 'Age_at_Scan'.
df_Liege_cleaned = utils.add_age_groups(df_Liege_cleaned)

# add a column of Group which represents the group of age groups and SEX groups
df_Liege_cleaned = utils.add_groups_age_sex(df_Liege_cleaned)

print("Data type of SEX column before conversion:", df_Liege_cleaned['SEX'].dtype)
# Convert to int if it's not already an int
if df_Liege_cleaned['SEX'].dtype != 'int64':
    df_Liege_cleaned['SEX'] = df_Liege_cleaned['SEX'].astype(int)
print("Data type of SEX column after conversion:", df_Liege_cleaned['SEX'].dtype)

# %%
# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Liege_cleaned[Sleep] = df_Liege_cleaned[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Liege_cleaned[Subcortical] = df_Liege_cleaned[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_cleaned[Thickness_DK] = df_Liege_cleaned[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_cleaned[Thickness_Schaefer] = df_Liege_cleaned[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_cleaned[Area_DK] = df_Liege_cleaned[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_cleaned[Area_Schaefer] = df_Liege_cleaned[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_cleaned[Subcortical] = df_Liege_cleaned[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Define feature lists
"""
Sleep = ['PSG_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Dur', 'Self_Sleep_Eff', 'Depression_score']
Cov = ['Age_at_Scan', 'SEX', 'BMI']
TIV = 'EstimatedTotalIntraCranialVol'

columns = df_Liege_cleaned.columns.tolist()
Thickness_DK = columns[columns.index('lh_bankssts_thickness'):columns.index('rh_insula_thickness')+1]
Thickness_Schaefer = columns[columns.index('LH_Vis_1_thickness'):columns.index('RH_Default_pCunPCC_9_thickness')+1]
Area_DK = columns[columns.index('lh_bankssts_area'):columns.index('rh_insula_area')+1]
Area_Schaefer = columns[columns.index('LH_Vis_1_area'):columns.index('RH_Default_pCunPCC_9_area')+1]
Subcortical = columns[columns.index('Left-Lateral-Ventricle'):columns.index('CC_Anterior')+1]

targets = ['Stroop_Test', 'Memory_Test']

# %%
"""
Define X and y
"""
X_dict = {
    'Sleep': Sleep,
    'Cov': Cov,
    'Sleep_Cov': Sleep + Cov,
    'Sleep_Cov_Brain': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_Cov_CT': Sleep + Cov + Thickness_DK + Thickness_Schaefer,
    'Sleep_Cov_SA': Sleep + Cov + Area_DK + Area_Schaefer,
    'Sleep_Cov_Subcor': Sleep + Cov + Subcortical,
    'Cov_Brain': Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Cov_CT': Cov + Thickness_DK + Thickness_Schaefer,
    'Cov_SA': Cov + Area_DK + Area_Schaefer,
    'Cov_Subcor': Cov + Subcortical,
    'Sleep_Shuffle_Cov': Sleep + Cov,
    'Sleep_Cov_Subcor_Shuffle': Sleep + Cov + Subcortical,
    'Cov_Brain_Shuffle': Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Cov_Subcor_Shuffle': Cov + Subcortical
}

y_dict = {
    'Stroop': 'Stroop_Test',
    'Memory': 'Memory_Test',
    '1-back': '1-back_acc',
    '2-back': '2-back_acc',
    '3-back': '3-back_acc'
}

# %%
print(f"\nStart model training")
case_results_path = results_path + f"{feature_comb}/"
if not os.path.exists(case_results_path):
    os.makedirs(case_results_path)

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_Liege_cleaned
df_test = None
save_path = case_results_path

ml_pipe_instance = MLPipeline(X, y, df_train, df_test, num_cores, save_path)
scores, model = ml_pipe_instance.ridge_rg_pipe(stratified_label='Group_Age_SEX')

print(f"Model training completed.\n")

# %%
# save scores and model
scores.to_csv(case_results_path + 'scores.csv')
from joblib import dump
dump(model, case_results_path + 'model.joblib')

# %%
print("\nModel performance:")
print("======================================")
print(f"Average CV Train MAE: {-scores['train_neg_mean_absolute_error'].mean():.4f}")
print(f"Average CV Test MAE: {-scores['test_neg_mean_absolute_error'].mean():.4f}")
print(f"Average CV Train RMSE: {-scores['train_neg_root_mean_squared_error'].mean():.4f}")
print(f"Average CV Test RMSE: {-scores['test_neg_root_mean_squared_error'].mean():.4f}")
print(f"Average CV Train R2: {scores['train_r2'].mean():.4f}")
print(f"Average CV Test R2: {scores['test_r2'].mean():.4f}")
print(f"Average CV Train Pearson r: {scores['train_r_corr'].mean():.4f}")
print(f"Average CV Test Pearson r: {scores['test_r_corr'].mean():.4f}")
print("======================================\n")

# create a dataframe to store the results
results_df = pd.DataFrame({
    'Model': ['Ridge Regression'],
    'Feature Combination': [feature_comb],
    'Target': [target],
    'Average CV Train MAE': [-scores['train_neg_mean_absolute_error'].mean()],
    'Average CV Test MAE': [-scores['test_neg_mean_absolute_error'].mean()],
    'Average CV Train RMSE': [-scores['train_neg_root_mean_squared_error'].mean()],
    'Average CV Test RMSE': [-scores['test_neg_root_mean_squared_error'].mean()],
    'Average CV Train R2': [scores['train_r2'].mean()],
    'Average CV Test R2': [scores['test_r2'].mean()],
    'Average CV Train Pearson r': [scores['train_r_corr'].mean()],
    'Average CV Test Pearson r': [scores['test_r_corr'].mean()]
})

results_df.to_csv(case_results_path + 'results.csv')

# %%
'''
SHAP explanation
'''
print("\nStart SHAP explanation")
import shap
import pickle

SHAP_path = case_results_path + 'SHAP/'
if not os.path.exists(SHAP_path):
    os.makedirs(SHAP_path)

explainer_train_set = shap.explainers.Permutation(model.predict, df_Liege_cleaned[X], max_evals=2000)

# %%
# explanation for the training set
explanation_train_set_train = explainer_train_set(df_Liege_cleaned[X])
with open(SHAP_path + 'explainer_train_set_train.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_train, f)

ml_pipe_instance.shap_explain(explanation_train_set_train, SHAP_path, 'train_set_train')

print("SHAP explanation completed.\n")

# %%
print(f"\nDone! Ridge regression pipeline for {target} prediction by {feature_comb} completed successfully.")
# print Date and Time
from datetime import datetime
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
