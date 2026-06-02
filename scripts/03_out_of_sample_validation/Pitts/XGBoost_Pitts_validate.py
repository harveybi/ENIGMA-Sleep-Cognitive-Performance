import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/lib/')  # change to the path where the lib folder is located
import utils
import XGBoost_pipeline
from XGBoost_pipeline import AdMLPipeline
import xgboost as xgb

import argparse
import pandas as pd
import shap
import pickle
from joblib import load
from datetime import datetime

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# reload XGBoost_pipeline
import importlib
importlib.reload(XGBoost_pipeline)
from XGBoost_pipeline import AdMLPipeline

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='XGBoost, out-of-sample validation')
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
    - Executive: Executive_Functioning_Test
    - Memory_Letter: Memory_Letter_Test
    - Memory_Spatial: Memory_Spatial_Test
    - Executive_rgo_age: Executive_Functioning_rgo_age
    - Memory_Letter_rgo_age: Memory_Letter_rgo_age
    - Memory_Spatial_rgo_age: Memory_Spatial_rgo_age
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target

print(f"\nStarting XGBoost out-of-sample validation pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'  # where the data is saved
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results/XGBoost/Pitts/{target}/{feature_comb}/'  # where the results will be saved

if target == 'Executive':
    trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/XGBoost/Stroop/{feature_comb}/'  # where the trained models are saved
if target == 'Executive_rgo_age':
    trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/XGBoost/Stroop_rgo_age/{feature_comb}/'  # where the trained models are saved
if target == 'Memory_Letter' or target == 'Memory_Spatial':
    trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/XGBoost/Memory/{feature_comb}/'  # where the trained models
if target == 'Memory_Letter_rgo_age' or target == 'Memory_Spatial_rgo_age':
    trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/XGBoost/Memory_rgo_age/{feature_comb}/'  # where the trained models

df_Pitts = pd.read_csv(data_save_path + 'Pitts_dataset_renamed_target_cleaned.csv')
df_Pitts = df_Pitts.dropna(subset=['Executive_Functioning_Test', 'Memory_Letter_Test', 'Memory_Spatial_Test'])

# print range of Stroop
print(f"Range of Executive_Functioning_Test: {df_Pitts['Executive_Functioning_Test'].min()} - {df_Pitts['Executive_Functioning_Test'].max()}")
print(f"Range of Memory_Letter_Test: {df_Pitts['Memory_Letter_Test'].min()} - {df_Pitts['Memory_Letter_Test'].max()}")
print(f"Range of Memory_Spatial_Test: {df_Pitts['Memory_Spatial_Test'].min()} - {df_Pitts['Memory_Spatial_Test'].max()}")

# %%
# # for column 'Executive_Functioning', 'Memory_Test1', 'Memory_Test2' in df_Pitts, rename 'Executive_Functioning' to 'Executive_Functioning_Test', rename 'Memory_Test1' to 'Memory_Letter_Test', rename 'Memory_Test2' to 'Memory_Spatial_Test'
# df_Pitts = df_Pitts.rename(columns={'Executive_Functioning': 'Executive_Functioning_Test', 'Memory_Test1': 'Memory_Letter_Test', 'Memory_Test2': 'Memory_Spatial_Test'})
#
# # save back
# df_Pitts.to_csv(data_save_path + 'Pitts_dataset_renamed_target_cleaned.csv', index=False)

# create a APOE4 column, which all value is nan
df_Pitts['APOE4'] = None

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
targets = feature_lists[9]

sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']

"""
Data preprocessing for Pitts dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Pitts_ml = utils.convert_units(df_Pitts, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_Pitts_ml[Thickness_DK] = df_Pitts_ml[Thickness_DK].div(df_Pitts_ml[Thickness_DK].sum(axis=1), axis=0)
df_Pitts_ml[Thickness_Schaefer] = df_Pitts_ml[Thickness_Schaefer].div(df_Pitts_ml[Thickness_Schaefer].sum(axis=1),
                                                                      axis=0)
df_Pitts_ml[Area_DK] = df_Pitts_ml[Area_DK].div(df_Pitts_ml[Area_DK].sum(axis=1), axis=0)
df_Pitts_ml[Area_Schaefer] = df_Pitts_ml[Area_Schaefer].div(df_Pitts_ml[Area_Schaefer].sum(axis=1), axis=0)
df_Pitts_ml[Subcortical] = df_Pitts_ml[Subcortical].div(df_Pitts_ml['EstimatedTotalIntraCranialVol'], axis=0)

# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Pitts_ml[Sleep] = df_Pitts_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Pitts_ml[Subcortical] = df_Pitts_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_Pitts_ml[Sleep] = df_Pitts_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_Pitts_ml[Sleep] = df_Pitts_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Pitts_ml[Thickness_DK] = df_Pitts_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Pitts_ml[Thickness_Schaefer] = df_Pitts_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Pitts_ml[Area_DK] = df_Pitts_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Pitts_ml[Area_Schaefer] = df_Pitts_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Pitts_ml[Subcortical] = df_Pitts_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Pitts_ml[Subcortical] = df_Pitts_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_Pitts_ml[APOE4] = df_Pitts_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_Pitts_ml[Thickness_DK] = df_Pitts_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Pitts_ml[Thickness_Schaefer] = df_Pitts_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Pitts_ml[Area_DK] = df_Pitts_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Pitts_ml[Area_Schaefer] = df_Pitts_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Pitts_ml[Subcortical] = df_Pitts_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# convert data types of columns. 'SEX' and 'APOE4' convert to object. Sleep, 'Age_at_Scan', 'BMI', TIV, and all imaging data to float64
columns_to_object = ['SEX', 'APOE4']
df_Pitts_ml[columns_to_object] = df_Pitts_ml[columns_to_object].astype('category')

# Convert specified columns to 'float64' data type
columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical
df_Pitts_ml[columns_to_float] = df_Pitts_ml[columns_to_float].astype('float64')

# %%
"""
Define X and y
"""
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
    'Executive': 'Executive_Functioning_Test',
    'Memory_Letter': 'Memory_Letter_Test',
    'Memory_Spatial': 'Memory_Spatial_Test',
    'Executive_rgo_age': 'Executive_Functioning_rgo_age',
    'Memory_Letter_rgo_age': 'Memory_Letter_rgo_age',
    'Memory_Spatial_rgo_age': 'Memory_Spatial_rgo_age',
}

print(f"\nLoad model")
model = xgb.XGBRegressor()
model.load_model(trained_models_path + f"xgboost_model.json")

X = X_dict[feature_comb]
y = y_dict[target]

df_test_Pitts = df_Pitts_ml
numeric_features = [feature for feature in X if df_test_Pitts[feature].dtype.name in ['int64', 'float64']]

# load the transformer
# preprocessors_saved_path = trained_models_path + 'whole_X_train_preprocessor.pkl'
# with open(preprocessors_saved_path, 'rb') as f:
#     whole_X_train_preprocessor = pickle.load(f)
# # if 'rgo_ago' in y, also load whole_age_poly and whole_age_lr
# if 'Age_at_Scan' in X and 'rgo_age' in y:
#     whole_age_poly_saved_path = trained_models_path + 'whole_age_poly.pkl'
#     with open(whole_age_poly_saved_path, 'rb') as f:
#         whole_age_poly = pickle.load(f)
#     whole_age_lr_saved_path = trained_models_path + 'whole_age_lr.pkl'
#     with open(whole_age_lr_saved_path, 'rb') as f:
#         whole_age_lr = pickle.load(f)
# else:
#     whole_age_poly = None
#     whole_age_lr = None
# Load the transformer using joblib
preprocessors_saved_path = trained_models_path + 'whole_X_train_preprocessor.pkl'
whole_X_train_preprocessor = load(preprocessors_saved_path)

# If 'rgo_age' in y, also load whole_age_poly and whole_age_lr
if 'Age_at_Scan' in X and 'rgo_age' in y:
    whole_age_poly_saved_path = trained_models_path + 'whole_age_poly.pkl'
    whole_age_poly = load(whole_age_poly_saved_path)

    whole_age_lr_saved_path = trained_models_path + 'whole_age_lr.pkl'
    whole_age_lr = load(whole_age_lr_saved_path)
else:
    whole_age_poly = None
    whole_age_lr = None

if not os.path.exists(results_path):
    os.makedirs(results_path)

ml_pipe_instance = AdMLPipeline(X, y, save_path=results_path, model=model)

test_mae_Pitts, test_rmse_Pitts, test_r2_Pitts, test_corr_Pitts, test_corr_p_Pitts, test_corr_spearman_Pitts, test_corr_spearman_p_Pitts, Pitts_pred, Pitts_true = ml_pipe_instance.test_perform_plot(df_test_Pitts, X_train_preprocessor=whole_X_train_preprocessor, age_poly=whole_age_poly, age_lr=whole_age_lr, fig_title=f'Pittsburgh', scatter_color='#96D3A1')

results_metrics_df = pd.DataFrame({
    'Model': ['XGBoost'],
    'Feature Combination': [feature_comb],
    'Target': [target],
    'Performance Test Set MAE Pitts': [test_mae_Pitts],
    'Performance Test Set RMSE Pitts': [test_rmse_Pitts],
    'Performance Test Set R2 Pitts': [test_r2_Pitts],
    'Performance Test Set Pearson r Pitts': [test_corr_Pitts],
    'Performance Test Set Pearson r p-value Pitts': [test_corr_p_Pitts],
    'Performance Test Set Spearman r Pitts': [test_corr_spearman_Pitts],
    'Performance Test Set Spearman r p-value Pitts': [test_corr_spearman_p_Pitts]
})

results_values_df = df_test_Pitts['Age_at_Scan']
results_values_df['Pitts_pred'] = Pitts_pred
results_values_df['Pitts_true'] = Pitts_true

# Save model metrics and performance
results_metrics_df.to_csv(results_path + 'results_metrics.csv', index=False)

# Save age, true values, and predicted values
results_values_df.to_csv(results_path + 'results_values.csv', index=False)

# print metrix dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Adjust the width to prevent truncation
pd.set_option('display.colheader_justify', 'left')  # Optional: Align columns for better readability
print(f"\n{results_metrics_df}\n")

print(f"\nDone! XGBoost pipeline for {target} prediction by {feature_comb} in Pitts completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
