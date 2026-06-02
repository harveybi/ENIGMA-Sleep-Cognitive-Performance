import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/lib/')  # change to the path where the lib folder is located
import utils
import AutoGluon_pipeline
from AutoGluon_pipeline import AdMLPipeline

import argparse
import pandas as pd
import numpy as np
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
    - Stroop: Stroop_Test
    - Memory_Digit: Memory_Digit_Test
    - Memory_Letter: Memory_Letter_Test
    - Stroop_rgo_age: Stroop_rgo_age
    - Memory_Digit_rgo_age: Memory_Digit_rgo_age
    - Memory_Letter_rgo_age: Memory_Letter_rgo_age
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target

print(f"\nStarting AutoGluon out-of-sample validation pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'  # where the data is saved
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results/AutoGluon/VETSA/{target}/{feature_comb}/'  # where the results will be saved

if 'Memory' in target:
    if 'rgo_age' in target:
        trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/Memory_rgo_age/{feature_comb}/'
    else:
        trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/Memory/{feature_comb}/'  # where the trained models are saved
else:
    trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/{target}/{feature_comb}/'  # where the trained models are saved

df_VETSA = pd.read_csv(data_save_path + 'VETSA_dataset_renamed_target_cleaned.csv')
df_VETSA = df_VETSA.dropna(subset=['Stroop_Test', 'Memory_Digit_Test', 'Memory_Letter_Test'])

# print range of Stroop
print(f"Range of Stroop_Test: {df_VETSA['Stroop_Test'].min()} - {df_VETSA['Stroop_Test'].max()}")
print(f"Range of Memory_Digit_Test: {df_VETSA['Memory_Digit_Test'].min()} - {df_VETSA['Memory_Digit_Test'].max()}")
print(f"Range of Memory_Letter_Test: {df_VETSA['Memory_Letter_Test'].min()} - {df_VETSA['Memory_Letter_Test'].max()}")

df_VETSA['PSG_Sleep_Dur'] = float('nan')
df_VETSA['PSG_Sleep_Eff'] = float('nan')

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
Data preprocessing for VETSA dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_VETSA_ml = utils.convert_units(df_VETSA, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_VETSA_ml[Thickness_DK] = df_VETSA_ml[Thickness_DK].div(df_VETSA_ml[Thickness_DK].sum(axis=1), axis=0)
df_VETSA_ml[Thickness_Schaefer] = df_VETSA_ml[Thickness_Schaefer].div(df_VETSA_ml[Thickness_Schaefer].sum(axis=1),
                                                                      axis=0)
df_VETSA_ml[Area_DK] = df_VETSA_ml[Area_DK].div(df_VETSA_ml[Area_DK].sum(axis=1), axis=0)
df_VETSA_ml[Area_Schaefer] = df_VETSA_ml[Area_Schaefer].div(df_VETSA_ml[Area_Schaefer].sum(axis=1), axis=0)
df_VETSA_ml[Subcortical] = df_VETSA_ml[Subcortical].div(df_VETSA_ml['EstimatedTotalIntraCranialVol'], axis=0)

# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_VETSA_ml[Sleep] = df_VETSA_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_VETSA_ml[Subcortical] = df_VETSA_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_VETSA_ml[Sleep] = df_VETSA_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_VETSA_ml[Sleep] = df_VETSA_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_VETSA_ml[Thickness_DK] = df_VETSA_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_VETSA_ml[Thickness_Schaefer] = df_VETSA_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_VETSA_ml[Area_DK] = df_VETSA_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_VETSA_ml[Area_Schaefer] = df_VETSA_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_VETSA_ml[Subcortical] = df_VETSA_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_VETSA_ml[Subcortical] = df_VETSA_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_VETSA_ml[APOE4] = df_VETSA_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_VETSA_ml[Thickness_DK] = df_VETSA_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_VETSA_ml[Thickness_Schaefer] = df_VETSA_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_VETSA_ml[Area_DK] = df_VETSA_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_VETSA_ml[Area_Schaefer] = df_VETSA_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_VETSA_ml[Subcortical] = df_VETSA_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)


# convert data types of columns. 'SEX' and 'APOE4' convert to object. Sleep, 'Age_at_Scan', 'BMI', TIV, and all imaging data to float64
columns_to_object = ['SEX', 'APOE4']
df_VETSA_ml[columns_to_object] = df_VETSA_ml[columns_to_object].astype('category')

# Convert specified columns to 'float64' data type
columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical
df_VETSA_ml[columns_to_float] = df_VETSA_ml[columns_to_float].astype('float64')

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
    'Stroop': 'Stroop_Test',
    'Memory_Digit': 'Memory_Digit_Test',
    'Memory_Letter': 'Memory_Letter_Test',
    'Stroop_rgo_age': 'Stroop_rgo_age',
    'Memory_Digit_rgo_age': 'Memory_Digit_rgo_age',
    'Memory_Letter_rgo_age': 'Memory_Letter_rgo_age'
}

print(f"\nLoad model")
# trained_models_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/Stroop/SHIP_Trend_no_DK/Sleep_Cov/'
model = TabularPredictor.load(trained_models_path + 'autogluon')
X = X_dict[feature_comb]
y = y_dict[target]

# %%
df_test_VETSA = df_VETSA_ml
numeric_features = [feature for feature in X if df_test_VETSA[feature].dtype.name in ['int64', 'float64']]

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
test_mae_VETSA, test_rmse_VETSA, test_r2_VETSA, test_corr_VETSA, test_corr_p_VETSA, test_corr_spearman_VETSA, test_corr_spearman_p_VETSA, VETSA_pred, VETSA_true = ml_pipe_instance.test_perform_plot(df_test_VETSA, age_poly=whole_age_poly, age_lr=whole_age_lr, fig_title=f'San Diego', scatter_color='#E4BDE5')

results_metrics_df = pd.DataFrame({
    'Model': ['AutoGluon'],
    'Feature Combination': [feature_comb],
    'Target': [target],
    'Performance Test Set MAE VETSA': [test_mae_VETSA],
    'Performance Test Set RMSE VETSA': [test_rmse_VETSA],
    'Performance Test Set R2 VETSA': [test_r2_VETSA],
    'Performance Test Set Pearson r VETSA': [test_corr_VETSA],
    'Performance Test Set Pearson r p-value VETSA': [test_corr_p_VETSA],
    'Performance Test Set Spearman r VETSA': [test_corr_spearman_VETSA],
    'Performance Test Set Spearman r p-value VETSA': [test_corr_spearman_p_VETSA]
})

# %%
# Fix the creation of results_values_df to avoid SettingWithCopyError
results_values_df = df_test_VETSA[['Age_at_Scan']].copy()  # Explicitly create a new DataFrame
# results_values_df['VETSA_pred'] = VETSA_pred  # Add predictions
results_values_df["VETSA_pred"] = np.expm1(VETSA_pred)  # for Sleep_Cov, Stroop
results_values_df['VETSA_true'] = VETSA_true  # Add true values

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

print(f"\nDone! AutoGluon pipeline for {target} prediction by {feature_comb} in VETSA completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
