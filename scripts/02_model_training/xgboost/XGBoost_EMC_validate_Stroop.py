import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import XGBoost_pipeline
from XGBoost_pipeline import AdMLPipeline
import xgboost as xgb

import argparse
import pandas as pd
import shap
import pickle
from datetime import datetime

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='XGBoost, out-of-sample validation')
parser.add_argument('feature_comb', type=str, help='''Name of the feature combination to be used.
    Available combinations are:
    - Sleep: Uses sleep features only.
    - Cov: Uses covariates only.
    - Sleep_Cov: Uses sleep features combined with covariates.
    - Sleep_Shuffle_Cov: Uses sleep features (shuffled) and covariates.
''')

parser.add_argument('target', type=str, help='''Name of the target to be predicted.
    Available targets are:
    - Stroop: Stroop_Test
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target

print(f"\nStarting XGBoost out-of-sample validation pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'  # where the data is saved
trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/{target}/SHIP_Trend/'  # where the trained models are saved
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/{target}/EMC/'  # where the results will be saved

# %%
df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed.csv')

# for df_SHIP, df_Liege, df_Liege_COF, df_Liege_COGNAP, remove subjects with missing values in 'Stroop_Test' and 'Memory_Test'
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test'])
df_Liege = df_Liege.dropna(subset=['Stroop_Test'])

# %%
"""
Define feature lists
"""
# load the feature lists from feature_lists.pkl
feature_lists_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/EMC_validation/feature_lists.pkl'
with open(feature_lists_path, 'rb') as f:
    feature_lists = pickle.load(f)

Sleep = feature_lists[0]
Cov = feature_lists[1]
TIV = feature_lists[2]
Thickness_DK = feature_lists[3]
Thickness_Schaefer = feature_lists[4]
Area_DK = feature_lists[5]
Area_Schaefer = feature_lists[6]
Subcortical = feature_lists[7]

# %%
"""
Data preprocessing for SHIP_Trend dataset
1. Convert sleep measurements units
2. Add group based on age and sex
3. Brain correction by brain size using internal data normalisation
4. NAI_Wordlist_Test transfer to accuracy
5. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']
df_SHIP_ml = utils.convert_units(df_SHIP, sleep_dur_cols, sleep_eff_cols)

# add a column of 'Age_Group' after 'Age_at_Scan'.
df_SHIP_ml = utils.add_age_groups(df_SHIP_ml)
# add a column of Group which represents the group of age groups and SEX groups
df_SHIP_ml = utils.add_groups_age_sex(df_SHIP_ml)

# Brain correction by brain size using internal data normalisation
df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].div(df_SHIP_ml[Thickness_DK].sum(axis=1), axis=0)
df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].div(df_SHIP_ml[Thickness_Schaefer].sum(axis=1), axis=0)
df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].div(df_SHIP_ml[Area_DK].sum(axis=1), axis=0)
df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].div(df_SHIP_ml[Area_Schaefer].sum(axis=1), axis=0)
df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].div(df_SHIP_ml['EstimatedTotalIntraCranialVol'], axis=0)

# NAI_Wordlist_Test transfer to accuracy
df_SHIP_ml['Memory_Test'] = df_SHIP_ml['Memory_Test'].apply(lambda x: x / 16) * 100

# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Data preprocessing for Liege dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Liege_ml = utils.convert_units(df_Liege, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_Liege_ml[Thickness_DK] = df_Liege_ml[Thickness_DK].div(df_Liege_ml[Thickness_DK].sum(axis=1), axis=0)
df_Liege_ml[Thickness_Schaefer] = df_Liege_ml[Thickness_Schaefer].div(df_Liege_ml[Thickness_Schaefer].sum(axis=1),
                                                                      axis=0)
df_Liege_ml[Area_DK] = df_Liege_ml[Area_DK].div(df_Liege_ml[Area_DK].sum(axis=1), axis=0)
df_Liege_ml[Area_Schaefer] = df_Liege_ml[Area_Schaefer].div(df_Liege_ml[Area_Schaefer].sum(axis=1), axis=0)
df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].div(df_Liege_ml['EstimatedTotalIntraCranialVol'], axis=0)

df_Liege_ml['Memory_Test'] = df_Liege_ml['Memory_Test'] * 100
df_Liege_ml['1-back_acc'] = df_Liege_ml['1-back_acc'] * 100
df_Liege_ml['2-back_acc'] = df_Liege_ml['2-back_acc'] * 100
df_Liege_ml['3-back_acc'] = df_Liege_ml['3-back_acc'] * 100

# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Liege_ml[Sleep] = df_Liege_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_Liege_ml[Sleep] = df_Liege_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_Liege_ml[Sleep] = df_Liege_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_ml[Thickness_DK] = df_Liege_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Thickness_Schaefer] = df_Liege_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Area_DK] = df_Liege_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Area_Schaefer] = df_Liege_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Define X and y
"""
X_dict = {
    'Sleep': Sleep,
    'Cov': Cov,
    'Sleep_Cov': Sleep + Cov,
    'Sleep_Shuffle_Cov': Sleep + Cov
}

y_dict = {
    'Stroop': 'Stroop_Test'
}

# %%
print(f"\nStart model training")
case_results_path = results_path + f"{feature_comb}/"
if not os.path.exists(case_results_path):
    os.makedirs(case_results_path)

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml

# X, y, df_train=None, num_cores=None, save_path=None
ml_pipe_instance = AdMLPipeline(X, y, df_train=df_train, num_cores=num_cores, save_path=case_results_path)

scores, model = ml_pipe_instance.xgboost_pipe(stratified_label='Group_Age_SEX')

print(f"\nModel training completed.\n")

# %%
# save score and model
scores.to_csv(case_results_path + 'scores.csv')

model_save_path = os.path.join(case_results_path, "xgboost_model.json")
# Save the trained XGBoost model
model.save_model(model_save_path)

# %%
df_test_Liege = df_Liege_ml
y_test_Liege = df_test_Liege[y].values.reshape(-1)
test_mae_Liege, test_rmse_Liege, test_r2_Liege, test_corr_Liege, test_corr_spearman_Liege = ml_pipe_instance.test_perform_plot(df_test_Liege, y_test_Liege, 'Test Set')

df_test_Liege_COF = df_Liege_COF_ml
y_test_Liege_COF = df_test_Liege_COF[y].values.reshape(-1)
test_mae_Liege_COF, test_rmse_Liege_COF, test_r2_Liege_COF, test_corr_Liege_COF, test_corr_spearman_Liege_COF = ml_pipe_instance.test_perform_plot(df_test_Liege_COF, y_test_Liege_COF, 'Test Set Liege COF')

df_test_Liege_COGNAP = df_Liege_COGNAP_ml
y_test_Liege_COGNAP = df_test_Liege_COGNAP[y].values.reshape(-1)
test_mae_Liege_COGNAP, test_rmse_Liege_COGNAP, test_r2_Liege_COGNAP, test_corr_Liege_COGNAP, test_corr_spearman_Liege_COGNAP = ml_pipe_instance.test_perform_plot(df_test_Liege_COGNAP, y_test_Liege_COGNAP, 'Test Set Liege COGNAP')

df_test_Liege_COF_COGNAP = pd.concat([df_test_Liege_COF, df_test_Liege_COGNAP], ignore_index=True)
y_test_Liege_COF_COGNAP = df_test_Liege_COF_COGNAP[y].values.reshape(-1)
test_mae_Liege_COF_COGNAP, test_rmse_Liege_COF_COGNAP, test_r2_Liege_COF_COGNAP, test_corr_Liege_COF_COGNAP, test_corr_spearman_Liege_COF_COGNAP = ml_pipe_instance.test_perform_plot(df_test_Liege_COF_COGNAP, y_test_Liege_COF_COGNAP, 'Test Set Liege COF COGNAP')

if target == 'Memory':
    df_test_KI = df_KI_ml
    y_test_KI = df_test_KI[y].values.reshape(-1)
    test_mae_KI, test_rmse_KI, test_r2_KI, test_corr_KI, test_corr_spearman_KI = ml_pipe_instance.test_perform_plot(df_test_KI, y_test_KI, 'Test Set KI')

    print("\nModel performance:")
    print("======================================")
    print(f"Average CV Train MAE: nan:")
    print(f"Average CV Test MAE: {-scores['test_neg_mean_absolute_error'].mean():.4f}")
    print(f"Performance Test Set MAE Liege: {test_mae_Liege:.4f}")
    print(f"Performance Test Set MAE Liege COF: {test_mae_Liege_COF:.4f}")
    print(f"Performance Test Set MAE Liege COGNAP: {test_mae_Liege_COGNAP:.4f}")
    print(f"Performance Test Set MAE Liege COF_COGNAP: {test_mae_Liege_COF_COGNAP:.4f}")
    print(f"Performance Test Set MAE KI: {test_mae_KI:.4f}")
    print(f"Average CV Train RMSE: nan")
    print(f"Average CV Test RMSE: {-scores['test_neg_root_mean_squared_error'].mean():.4f}")
    print(f"Performance Test Set RMSE Liege: {test_rmse_Liege:.4f}")
    print(f"Performance Test Set RMSE Liege COF: {test_rmse_Liege_COF:.4f}")
    print(f"Performance Test Set RMSE Liege COGNAP: {test_rmse_Liege_COGNAP:.4f}")
    print(f"Performance Test Set RMSE Liege COF_COGNAP: {test_rmse_Liege_COF_COGNAP:.4f}")
    print(f"Performance Test Set RMSE KI: {test_rmse_KI:.4f}")
    print(f"Average CV Train R2: nan")
    print(f"Average CV Test R2: {scores['test_r2'].mean():.4f}")
    print(f"Performance Test Set R2 Liege: {test_r2_Liege:.4f}")
    print(f"Performance Test Set R2 Liege COF: {test_r2_Liege_COF:.4f}")
    print(f"Performance Test Set R2 Liege COGNAP: {test_r2_Liege_COGNAP:.4f}")
    print(f"Performance Test Set R2 Liege COF_COGNAP: {test_r2_Liege_COF_COGNAP:.4f}")
    print(f"Performance Test Set R2 KI: {test_r2_KI:.4f}")
    print(f"Average CV Train Pearson r: nan")
    print(f"Average CV Test Pearson r: {scores['test_r_corr'].mean():.4f}")
    print(f"Performance Test Set Pearson r Liege: {test_corr_Liege:.4f}")
    print(f"Performance Test Set Pearson r Liege COF: {test_corr_Liege_COF:.4f}")
    print(f"Performance Test Set Pearson r Liege COGNAP: {test_corr_Liege_COGNAP:.4f}")
    print(f"Performance Test Set Pearson r Liege COF_COGNAP: {test_corr_Liege_COF_COGNAP:.4f}")
    print(f"Performance Test Set Pearson r KI: {test_corr_KI:.4f}")
    print(f"Average CV Train Spearman r: nan")
    print(f"Average CV Test Spearman r: {scores['test_spearmanr'].mean():.4f}")
    print(f"Performance Test Set Spearman r Liege: {test_corr_spearman_Liege:.4f}")
    print(f"Performance Test Set Spearman r Liege COF: {test_corr_spearman_Liege_COF:.4f}")
    print(f"Performance Test Set Spearman r Liege COGNAP: {test_corr_spearman_Liege_COGNAP:.4f}")
    print(f"Performance Test Set Spearman r Liege COF_COGNAP: {test_corr_spearman_Liege_COF_COGNAP:.4f}")
    print(f"Performance Test Set Spearman r KI: {test_corr_spearman_KI:.4f}")
    print("======================================\n")

    # create a dataframe to store the results
    results_df = pd.DataFrame({
        'Model': ['XGBoost'],
        'Feature Combination': [feature_comb],
        'Target': [target],
        'Average CV Train MAE': 'nan',
        'Average CV Test MAE': [-scores['test_neg_mean_absolute_error'].mean()],
        'Performance Test Set MAE Liege': [test_mae_Liege],
        'Performance Test Set MAE Liege COF': [test_mae_Liege_COF],
        'Performance Test Set MAE Liege COGNAP': [test_mae_Liege_COGNAP],
        'Performance Test Set MAE Liege COF_COGNAP': [test_mae_Liege_COF_COGNAP],
        'Performance Test Set MAE KI': [test_mae_KI],
        'Average CV Train RMSE': 'nan',
        'Average CV Test RMSE': [-scores['test_neg_root_mean_squared_error'].mean()],
        'Performance Test Set RMSE Liege': [test_rmse_Liege],
        'Performance Test Set RMSE Liege COF': [test_rmse_Liege_COF],
        'Performance Test Set RMSE Liege COGNAP': [test_rmse_Liege_COGNAP],
        'Performance Test Set RMSE Liege COF_COGNAP': [test_rmse_Liege_COF_COGNAP],
        'Performance Test Set RMSE KI': [test_rmse_KI],
        'Average CV Train R2': 'nan',
        'Average CV Test R2': [scores['test_r2'].mean()],
        'Performance Test Set R2 Liege': [test_r2_Liege],
        'Performance Test Set R2 Liege COF': [test_r2_Liege_COF],
        'Performance Test Set R2 Liege COGNAP': [test_r2_Liege_COGNAP],
        'Performance Test Set R2 Liege COF_COGNAP': [test_r2_Liege_COF_COGNAP],
        'Performance Test Set R2 KI': [test_r2_KI],
        'Average CV Train Pearson r': 'nan',
        'Average CV Test Pearson r': [scores['test_r_corr'].mean()],
        'Performance Test Set Pearson r Liege': [test_corr_Liege],
        'Performance Test Set Pearson r Liege COF': [test_corr_Liege_COF],
        'Performance Test Set Pearson r Liege COGNAP': [test_corr_Liege_COGNAP],
        'Performance Test Set Pearson r Liege COF_COGNAP': [test_corr_Liege_COF_COGNAP],
        'Performance Test Set Pearson r KI': [test_corr_KI],
        'Average CV Train Spearman r': 'nan',
        'Average CV Test Spearman r': [scores['test_spearmanr'].mean()],
        'Performance Test Set Spearman r Liege': [test_corr_spearman_Liege],
        'Performance Test Set Spearman r Liege COF': [test_corr_spearman_Liege_COF],
        'Performance Test Set Spearman r Liege COGNAP': [test_corr_spearman_Liege_COGNAP],
        'Performance Test Set Spearman r Liege COF_COGNAP': [test_corr_spearman_Liege_COF_COGNAP],
        'Performance Test Set Spearman r KI': [test_corr_spearman_KI]
    })

    results_df.to_csv(case_results_path + 'results.csv')

else:
    print("\nModel performance:")
    print("======================================")
    print(f"Average CV Train MAE: nan:")
    print(f"Average CV Test MAE: {-scores['test_neg_mean_absolute_error'].mean():.4f}")
    print(f"Performance Test Set MAE Liege: {test_mae_Liege:.4f}")
    print(f"Performance Test Set MAE Liege COF: {test_mae_Liege_COF:.4f}")
    print(f"Performance Test Set MAE Liege COGNAP: {test_mae_Liege_COGNAP:.4f}")
    print(f"Performance Test Set MAE Liege COF_COGNAP: {test_mae_Liege_COF_COGNAP:.4f}")
    print(f"Average CV Train RMSE: nan")
    print(f"Average CV Test RMSE: {-scores['test_neg_root_mean_squared_error'].mean():.4f}")
    print(f"Performance Test Set RMSE Liege: {test_rmse_Liege:.4f}")
    print(f"Performance Test Set RMSE Liege COF: {test_rmse_Liege_COF:.4f}")
    print(f"Performance Test Set RMSE Liege COGNAP: {test_rmse_Liege_COGNAP:.4f}")
    print(f"Performance Test Set RMSE Liege COF_COGNAP: {test_rmse_Liege_COF_COGNAP:.4f}")
    print(f"Average CV Train R2: nan")
    print(f"Average CV Test R2: {scores['test_r2'].mean():.4f}")
    print(f"Performance Test Set R2 Liege: {test_r2_Liege:.4f}")
    print(f"Performance Test Set R2 Liege COF: {test_r2_Liege_COF:.4f}")
    print(f"Performance Test Set R2 Liege COGNAP: {test_r2_Liege_COGNAP:.4f}")
    print(f"Performance Test Set R2 Liege COF_COGNAP: {test_r2_Liege_COF_COGNAP:.4f}")
    print(f"Average CV Train Pearson r: nan")
    print(f"Average CV Test Pearson r: {scores['test_r_corr'].mean():.4f}")
    print(f"Performance Test Set Pearson r Liege: {test_corr_Liege:.4f}")
    print(f"Performance Test Set Pearson r Liege COF: {test_corr_Liege_COF:.4f}")
    print(f"Performance Test Set Pearson r Liege COGNAP: {test_corr_Liege_COGNAP:.4f}")
    print(f"Performance Test Set Pearson r Liege COF_COGNAP: {test_corr_Liege_COF_COGNAP:.4f}")
    print(f"Average CV Train Spearman r: nan")
    print(f"Average CV Test Spearman r: {scores['test_spearmanr'].mean():.4f}")
    print(f"Performance Test Set Spearman r Liege: {test_corr_spearman_Liege:.4f}")
    print(f"Performance Test Set Spearman r Liege COF: {test_corr_spearman_Liege_COF:.4f}")
    print(f"Performance Test Set Spearman r Liege COGNAP: {test_corr_spearman_Liege_COGNAP:.4f}")
    print(f"Performance Test Set Spearman r Liege COF_COGNAP: {test_corr_spearman_Liege_COF_COGNAP:.4f}")
    print("======================================\n")

    # create a dataframe to store the results
    results_df = pd.DataFrame({
        'Model': ['XGBoost'],
        'Feature Combination': [feature_comb],
        'Target': [target],
        'Average CV Train MAE': 'nan',
        'Average CV Test MAE': [-scores['test_neg_mean_absolute_error'].mean()],
        'Performance Test Set MAE Liege': [test_mae_Liege],
        'Performance Test Set MAE Liege COF': [test_mae_Liege_COF],
        'Performance Test Set MAE Liege COGNAP': [test_mae_Liege_COGNAP],
        'Performance Test Set MAE Liege COF_COGNAP': [test_mae_Liege_COF_COGNAP],
        'Average CV Train RMSE': 'nan',
        'Average CV Test RMSE': [-scores['test_neg_root_mean_squared_error'].mean()],
        'Performance Test Set RMSE Liege': [test_rmse_Liege],
        'Performance Test Set RMSE Liege COF': [test_rmse_Liege_COF],
        'Performance Test Set RMSE Liege COGNAP': [test_rmse_Liege_COGNAP],
        'Performance Test Set RMSE Liege COF_COGNAP': [test_rmse_Liege_COF_COGNAP],
        'Average CV Train R2': 'nan',
        'Average CV Test R2': [scores['test_r2'].mean()],
        'Performance Test Set R2 Liege': [test_r2_Liege],
        'Performance Test Set R2 Liege COF': [test_r2_Liege_COF],
        'Performance Test Set R2 Liege COGNAP': [test_r2_Liege_COGNAP],
        'Performance Test Set R2 Liege COF_COGNAP': [test_r2_Liege_COF_COGNAP],
        'Average CV Train Pearson r': 'nan',
        'Average CV Test Pearson r': [scores['test_r_corr'].mean()],
        'Performance Test Set Pearson r Liege': [test_corr_Liege],
        'Performance Test Set Pearson r Liege COF': [test_corr_Liege_COF],
        'Performance Test Set Pearson r Liege COGNAP': [test_corr_Liege_COGNAP],
        'Performance Test Set Pearson r Liege COF_COGNAP': [test_corr_Liege_COF_COGNAP],
        'Average CV Train Spearman r': 'nan',
        'Average CV Test Spearman r': [scores['test_spearmanr'].mean()],
        'Performance Test Set Spearman r Liege': [test_corr_spearman_Liege],
        'Performance Test Set Spearman r Liege COF': [test_corr_spearman_Liege_COF],
        'Performance Test Set Spearman r Liege COGNAP': [test_corr_spearman_Liege_COGNAP],
        'Performance Test Set Spearman r Liege COF_COGNAP': [test_corr_spearman_Liege_COF_COGNAP]
    })

    results_df.to_csv(case_results_path + 'results.csv')

# %%
print(f"\nDone! XGBoost pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
