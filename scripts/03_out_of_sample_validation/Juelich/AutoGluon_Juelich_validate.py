import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/lib/')  # change to the path where the lib folder is located
import utils
import AutoGluon_pipeline
from AutoGluon_pipeline import AdMLPipeline

import argparse
import pandas as pd
import shap
import pickle
from autogluon.tabular import TabularPredictor
from joblib import load
from datetime import datetime

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# reload AutoGluon_pipeline
import importlib
importlib.reload(AutoGluon_pipeline)
from AutoGluon_pipeline import AdMLPipeline

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='AutoGluon, out-of-sample validation')
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
    - Memory_Letter: Letter_Sensitivity_Test
    - Memory_Spatial: Spatial_Sensitivity_Test
    - Memory_Letter_rgo_age: Letter_Sensitivity_rgo_age
    - Memory_Spatial_rgo_age: Spatial_Sensitivity_rgo_age
''')

parser.add_argument('sess', type=str, help='''Name of the target to be predicted.
    Available targets are:
    - sess-1: SomnoSafe_1
    - sess-2: SomnoSafe_2
    - sess-SD: SomnoSafe_SD
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target
sess = args.sess

print(f"\nStarting AutoGluon out-of-sample validation pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'  # where the data is saved
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results/AutoGluon/Juelich/{sess}/{target}/{feature_comb}/'  # where the results will be saved

if target == 'Memory_Letter' or target == 'Memory_Spatial':
    trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/Memory/{feature_comb}/'  # where the trained models
if target == 'Memory_Letter_rgo_age' or target == 'Memory_Spatial_rgo_age':
    trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/Memory_rgo_age/{feature_comb}/'  # where the trained models

if sess == 'sess-1':
    df_Juelich = pd.read_csv(data_save_path + 'Juelich_1_all_renamed_target_cleaned.csv', index_col=0)
    df_Juelich = df_Juelich.dropna(subset=['Letter_Sensitivity_Test', 'Spatial_Sensitivity_Test'])
if sess == 'sess-2':
    df_Juelich = pd.read_csv(data_save_path + 'Juelich_2_all_renamed_target_cleaned.csv', index_col=0)
    df_Juelich = df_Juelich.dropna(subset=['Letter_Sensitivity_Test', 'Spatial_Sensitivity_Test'])
if sess == 'sess-SD':
    df_Juelich = pd.read_csv(data_save_path + 'Juelich_SD_all_renamed_target_cleaned.csv', index_col=0)
    df_Juelich = df_Juelich.dropna(subset=['Letter_Sensitivity_Test', 'Spatial_Sensitivity_Test'])

# print range of Stroop
print(f"Range of Letter_Sensitivity_Test: {df_Juelich['Letter_Sensitivity_Test'].min()} - {df_Juelich['Letter_Sensitivity_Test'].max()}")
print(f"Range of Spatial_Sensitivity_Test: {df_Juelich['Spatial_Sensitivity_Test'].min()} - {df_Juelich['Spatial_Sensitivity_Test'].max()}")

df_Juelich['APOE4'] = None

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

# %%
"""
Data preprocessing for Juelich dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Juelich_ml = utils.convert_units(df_Juelich, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_Juelich_ml[Thickness_DK] = df_Juelich_ml[Thickness_DK].div(df_Juelich_ml[Thickness_DK].sum(axis=1), axis=0)
df_Juelich_ml[Thickness_Schaefer] = df_Juelich_ml[Thickness_Schaefer].div(df_Juelich_ml[Thickness_Schaefer].sum(axis=1),
                                                                      axis=0)
df_Juelich_ml[Area_DK] = df_Juelich_ml[Area_DK].div(df_Juelich_ml[Area_DK].sum(axis=1), axis=0)
df_Juelich_ml[Area_Schaefer] = df_Juelich_ml[Area_Schaefer].div(df_Juelich_ml[Area_Schaefer].sum(axis=1), axis=0)

df_Juelich_ml[Subcortical] = df_Juelich_ml[Subcortical].div(df_Juelich_ml['EstimatedTotalIntraCranialVol'], axis=0)

# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Juelich_ml[Sleep] = df_Juelich_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Juelich_ml[Subcortical] = df_Juelich_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_Juelich_ml[Sleep] = df_Juelich_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_Juelich_ml[Sleep] = df_Juelich_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Juelich_ml[Thickness_DK] = df_Juelich_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Juelich_ml[Thickness_Schaefer] = df_Juelich_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Juelich_ml[Area_DK] = df_Juelich_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Juelich_ml[Area_Schaefer] = df_Juelich_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Juelich_ml[Subcortical] = df_Juelich_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Juelich_ml[Subcortical] = df_Juelich_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_Juelich_ml[APOE4] = df_Juelich_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_Juelich_ml[Thickness_DK] = df_Juelich_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Juelich_ml[Thickness_Schaefer] = df_Juelich_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Juelich_ml[Area_DK] = df_Juelich_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Juelich_ml[Area_Schaefer] = df_Juelich_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Juelich_ml[Subcortical] = df_Juelich_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)


# convert data types of columns. 'SEX' and 'APOE4' convert to object. Sleep, 'Age_at_Scan', 'BMI', TIV, and all imaging data to float64
columns_to_object = ['SEX', 'APOE4']
df_Juelich_ml[columns_to_object] = df_Juelich_ml[columns_to_object].astype('category')

# Convert specified columns to 'float64' data type
columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical
df_Juelich_ml[columns_to_float] = df_Juelich_ml[columns_to_float].astype('float64')

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
    'Memory_Letter': 'Letter_Sensitivity_Test',
    'Memory_Spatial': 'Spatial_Sensitivity_Test',
    'Memory_Letter_rgo_age': 'Letter_Sensitivity_rgo_age',
    'Memory_Spatial_rgo_age': 'Spatial_Sensitivity_rgo_age'
}

print(f"\nLoad model")
model = TabularPredictor.load(trained_models_path + 'autogluon')
X = X_dict[feature_comb]
y = y_dict[target]

# %%
df_test_Juelich = df_Juelich_ml
numeric_features = [feature for feature in X if df_test_Juelich[feature].dtype.name in ['int64', 'float64']]

# load the transformer
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

# %%
test_mae_Juelich, test_rmse_Juelich, test_r2_Juelich, test_corr_Juelich, test_corr_p_Juelich, test_corr_spearman_Juelich, test_corr_spearman_p_Juelich, Juelich_pred, Juelich_true = ml_pipe_instance.test_perform_plot(df_test_Juelich, age_poly=whole_age_poly, age_lr=whole_age_lr, fig_title=f'Jülich', scatter_color='#90A2A5')

results_metrics_df = pd.DataFrame({
    'Model': ['AutoGluon'],
    'Feature Combination': [feature_comb],
    'Target': [target],
    'Performance Test Set MAE Juelich': [test_mae_Juelich],
    'Performance Test Set RMSE Juelich': [test_rmse_Juelich],
    'Performance Test Set R2 Juelich': [test_r2_Juelich],
    'Performance Test Set Pearson r Juelich': [test_corr_Juelich],
    'Performance Test Set Pearson r p-value Juelich': [test_corr_p_Juelich],
    'Performance Test Set Spearman r Juelich': [test_corr_spearman_Juelich],
    'Performance Test Set Spearman r p-value Juelich': [test_corr_spearman_p_Juelich]
})

# Fix the creation of results_values_df to avoid SettingWithCopyError
results_values_df = df_test_Juelich[['Age_at_Scan']].copy()  # Explicitly create a new DataFrame
results_values_df['Juelich_pred'] = Juelich_pred  # Add predictions
results_values_df['Juelich_true'] = Juelich_true  # Add true values

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

print(f"\nDone! AutoGluon pipeline for {target} prediction by {feature_comb} in Juelich completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
