import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
# import AutoGluon_pipeline
# from AutoGluon_pipeline import AdMLPipeline
import AutoGluon_pipeline_GPU
from AutoGluon_pipeline_GPU import AdMLPipeline

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
parser = argparse.ArgumentParser(description='AutoGluon, SHIP_Trend dataset.')
parser.add_argument('feature_comb', type=str, help='''Name of the feature combination to be used.
    Available combinations are:
    - Sleep: Uses sleep features only.
    - Cov: Uses covariates only.
    - Brain: Uses brain imaging data only.
    - CT: Uses cortical thickness from imaging data only.
    - SA: Uses surface area from imaging data only.
    - Subcor: Uses subcortical volumes only.
    - Sleep_Cov: Uses sleep features combined with covariates.
    - Sleep_Cov_Brain: Uses sleep features, covariates, and brain imaging data.
    - Sleep_Cov_CT: Uses sleep features, covariates, and cortical thickness from imaging data.
    - Sleep_Cov_SA: Uses sleep features, covariates, and surface area from imaging data.
    - Sleep_Cov_Subcor: Uses sleep features, covariates, and subcortical volumes.
    - Sleep_Brain: Uses sleep features and brain imaging data.
    - Sleep_CT: Uses sleep features and cortical thickness from imaging data.
    - Sleep_SA: Uses sleep features and surface area from imaging data.
    - Sleep_Subcor: Uses sleep features and subcortical volumes.
    - Cov_Brain: Uses covariates combined with brain imaging data.
    - Cov_CT: Uses covariates and cortical thickness from imaging data.
    - Cov_SA: Uses covariates and surface area from imaging data.
    - Cov_Subcor: Uses covariates and subcortical volumes.
    - Sleep_Shuffle_Cov: Uses sleep features (shuffled) and covariates.
    - Sleep_Cov_Subcor_Shuffle: Uses sleep features, covariates, and subcortical volumes (shuffled).
    - Sleep_Shuffle_Cov_Subcor: Uses sleep features (shuffled), covariates, and subcortical volumes.
    - Sleep_Shuffle_Subcor: Uses sleep features (shuffled) and subcortical volumes.
    - Cov_Brain_Shuffle: Uses covariates and brain imaging data (shuffled).
    - Cov_Subcor_Shuffle: Uses covariates and subcortical volumes (shuffled).
    - Sleep_Cov_Brain_Shuffle: Uses sleep features, covariates, and brain imaging data (shuffled).
    - Sleep_APOE: Sleep + APOE
    - Sleep_APOE_Shuffle: Sleep + APOE (Shuffled)
    - Sleep_Cov_APOE: Sleep + Cov + APOE
    - Sleep_Cov_APOE_Shuffle: Sleep + Cov + APOE (Shuffled)
    - Sleep_Cov_Brain_APOE: Sleep + Cov + Brain + APOE
    - Sleep_Cov_Brain_APOE_Shuffle: Sleep + Cov + Brain + APOE (Shuffled)
''')

parser.add_argument('target', type=str, help='''Name of the target to be predicted.
    Available targets are:
    - Stroop: Stroop_Test
    - Memory: NAI_Wordlist_Test
    - Stroop_rgo_age: Stroop_rgo_age
    - Memory_rgo_age: Memory_rgo_age
''')

# Add num_cores as an optional argument with a default value
parser.add_argument('num_cores', type=int, default=1,
                    help='Number of cores to be used for parallel processing. Default is 1.')

parser.add_argument('num_gpus', type=int, default=1,)

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target
num_cores = args.num_cores
num_gpus = args.num_gpus

print(f"\nStarting AutoGluon pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/{target}/SHIP_Trend_no_DK/'

# %%
df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed.csv')
df_Liege_COF = pd.read_csv(data_save_path + 'Liege_COF_dataset_renamed.csv')
df_Liege_COGNAP = pd.read_csv(data_save_path + 'Liege_COGNAP_dataset_renamed.csv')
df_KI = pd.read_csv(data_save_path + 'KI_dataset_renamed.csv')

# %%
# for df_SHIP, df_Liege, df_Liege_COF, df_Liege_COGNAP, remove subjects with missing values in 'Stroop_Test' and 'Memory_Test'
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test', 'Memory_Test'])
df_Liege = df_Liege.dropna(subset=['Stroop_Test', 'Memory_Test'])
df_Liege_COF = df_Liege_COF.dropna(subset=['Stroop_Test', 'Memory_Test'])
df_Liege_COGNAP = df_Liege_COGNAP.dropna(subset=['Stroop_Test', 'Memory_Test'])

# for df_KI, remove subjects with missing values in 'Memory_Test'
df_KI = df_KI.dropna(subset=['Memory_Test'])

# %%
"""
Define feature lists
"""
Sleep = ['PSG_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Dur', 'Self_Sleep_Eff', 'Depression_score']
Cov = ['Age_at_Scan', 'SEX', 'BMI']
APOE4 = ['APOE4']
TIV = 'EstimatedTotalIntraCranialVol'

columns = df_SHIP.columns.tolist()
Thickness_DK = columns[columns.index('lh_bankssts_thickness'):columns.index('rh_insula_thickness') + 1]
Thickness_Schaefer = columns[columns.index('LH_Vis_1_thickness'):columns.index('RH_Default_pCunPCC_9_thickness') + 1]
Area_DK = columns[columns.index('lh_bankssts_area'):columns.index('rh_insula_area') + 1]
Area_Schaefer = columns[columns.index('LH_Vis_1_area'):columns.index('RH_Default_pCunPCC_9_area') + 1]
Subcortical = columns[columns.index('Left-Lateral-Ventricle'):columns.index('CC_Anterior') + 1]

targets = ['Stroop_Test', 'Memory_Test']

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
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_SHIP_ml[APOE4] = df_SHIP_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)

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
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_Liege_ml[Thickness_DK] = df_Liege_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Thickness_Schaefer] = df_Liege_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Area_DK] = df_Liege_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Area_Schaefer] = df_Liege_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_Liege_ml[APOE4] = df_Liege_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Data preprocessing for KI dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_KI_ml = utils.convert_units(df_KI, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_KI_ml[Thickness_DK] = df_KI_ml[Thickness_DK].div(df_KI_ml[Thickness_DK].sum(axis=1), axis=0)
df_KI_ml[Thickness_Schaefer] = df_KI_ml[Thickness_Schaefer].div(df_KI_ml[Thickness_Schaefer].sum(axis=1), axis=0)
df_KI_ml[Area_DK] = df_KI_ml[Area_DK].div(df_KI_ml[Area_DK].sum(axis=1), axis=0)
df_KI_ml[Area_Schaefer] = df_KI_ml[Area_Schaefer].div(df_KI_ml[Area_Schaefer].sum(axis=1), axis=0)
df_KI_ml[Subcortical] = df_KI_ml[Subcortical].div(df_KI_ml['EstimatedTotalIntraCranialVol'], axis=0)

# shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_KI_ml[Sleep] = df_KI_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_KI_ml[Subcortical] = df_KI_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_KI_ml[Sleep] = df_KI_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_KI_ml[Sleep] = df_KI_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_KI_ml[Thickness_DK] = df_KI_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Thickness_Schaefer] = df_KI_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Area_DK] = df_KI_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Area_Schaefer] = df_KI_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Subcortical] = df_KI_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_KI_ml[Subcortical] = df_KI_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_KI_ml[Thickness_DK] = df_KI_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Thickness_Schaefer] = df_KI_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Area_DK] = df_KI_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Area_Schaefer] = df_KI_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Subcortical] = df_KI_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_KI_ml[APOE4] = df_KI_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
# convert data types of columns. 'SEX' and 'APOE4' convert to object. Sleep, 'Age_at_Scan', 'BMI', TIV, and all imaging data to float64
columns_to_object = ['SEX', 'APOE4']
df_SHIP_ml[columns_to_object] = df_SHIP_ml[columns_to_object].astype('category')
df_Liege_ml[columns_to_object] = df_Liege_ml[columns_to_object].astype('category')
df_KI_ml[columns_to_object] = df_KI_ml[columns_to_object].astype('category')

# Convert specified columns to 'float64' data type
columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + targets
columns_to_float_KI = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + ['Memory_Test']
df_SHIP_ml[columns_to_float] = df_SHIP_ml[columns_to_float].astype('float64')
df_Liege_ml[columns_to_float] = df_Liege_ml[columns_to_float].astype('float64')
df_KI_ml[columns_to_float_KI] = df_KI_ml[columns_to_float_KI].astype('float64')

# %%
"""
Define X and y
"""
X_dict = {
    'Sleep': Sleep,
    'Cov': Cov,
    'Brain': Thickness_Schaefer + Area_Schaefer + Subcortical,
    'CT': Thickness_Schaefer,
    'SA': Area_Schaefer,
    'Subcor': Subcortical,
    'Sleep_Cov': Sleep + Cov,
    'Sleep_Cov_Brain': Sleep + Cov + Thickness_Schaefer + Area_Schaefer + Subcortical,
    'Sleep_Cov_CT': Sleep + Cov + Thickness_Schaefer,
    'Sleep_Cov_SA': Sleep + Cov + Area_Schaefer,
    'Sleep_Cov_Subcor': Sleep + Cov + Subcortical,
    'Sleep_Brain': Sleep + Thickness_Schaefer + Area_Schaefer + Subcortical,
    'Sleep_CT': Sleep + Thickness_Schaefer,
    'Sleep_SA': Sleep + Area_Schaefer,
    'Sleep_Subcor': Sleep + Subcortical,
    'Cov_Brain': Cov + Thickness_Schaefer + Area_Schaefer + Subcortical,
    'Cov_CT': Cov + Thickness_Schaefer,
    'Cov_SA': Cov + Area_Schaefer,
    'Cov_Subcor': Cov + Subcortical,
    'Sleep_Shuffle_Cov': Sleep + Cov,
    'Sleep_Cov_Subcor_Shuffle': Sleep + Cov + Subcortical,
    'Sleep_Shuffle_Cov_Subcor': Sleep + Cov + Subcortical,
    'Sleep_Shuffle_Subcor': Sleep + Subcortical,
    'Cov_Brain_Shuffle': Cov + Thickness_Schaefer + Area_Schaefer + Subcortical,
    'Cov_Subcor_Shuffle': Cov + Subcortical,
    'Sleep_Cov_Brain_Shuffle': Sleep + Cov + Thickness_Schaefer + Area_Schaefer + Subcortical,
    'Sleep_APOE': Sleep + APOE4,
    'Sleep_APOE_Shuffle': Sleep + APOE4,
    'Sleep_Cov_APOE': Sleep + Cov + APOE4,
    'Sleep_Cov_APOE_Shuffle': Sleep + Cov + APOE4,
    'Sleep_Cov_Brain_APOE': Sleep + Cov + Thickness_Schaefer + Area_Schaefer + Subcortical + APOE4,
    'Sleep_Cov_Brain_APOE_Shuffle': Sleep + Cov + Thickness_Schaefer + Area_Schaefer + Subcortical + APOE4,
}

y_dict = {
    'Stroop': 'Stroop_Test',
    'Memory': 'Memory_Test',
    'Stroop_rgo_age': 'Stroop_Test_rgo_age',
    'Memory_rgo_age': 'Memory_Test_rgo_age'
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
ml_pipe_instance = AdMLPipeline(X, y, df_train, num_cores, num_gpus, case_results_path)

model = ml_pipe_instance.autogluon_pipe()

# TODO: In advanced_ml_pipeline, add a function to get cross-validation results
# scores = ml_pipe_instance.run_autogluon_cross_val_iteration(stratified_label='Group_Age_SEX')

print(f"\nModel training completed.\n")

# %%
# save scores
# scores.to_csv(case_results_path + 'scores.csv', index=False)
results = model.fit_summary()
# print full of results
import pprint
pprint.pprint(results)

# %%
df_test_Liege = df_Liege_ml
# y_test_Liege = df_test_Liege[y].values.reshape(-1)
score_var_Liege, test_mae_Liege, test_rmse_Liege, test_r2_Liege, test_corr_Liege, test_corr_spearman_Liege = ml_pipe_instance.test_perform_plot(df_test_Liege, 'Test Set')

if target == 'Memory' or target == 'Memory_rgo_age':
    df_test_KI = df_KI_ml
    # y_test_KI = df_test_KI[y].values.reshape(-1)
    score_var_KI, test_mae_KI, test_rmse_KI, test_r2_KI, test_corr_KI, test_corr_spearman_KI = ml_pipe_instance.test_perform_plot(df_test_KI, 'Test Set KI')

    print("\nModel performance:")
    print("======================================")
    print(f"Average CV Test MAE: nan:")
    # print(f"Average CV Test MAE: {-scores['test_neg_mean_absolute_error'].mean():.4f}")
    print(f"Performance Test Set MAE Liege: {test_mae_Liege:.4f}")
    print(f"Performance Test Set MAE KI: {test_mae_KI:.4f}")
    print(f"Average CV Test RMSE: nan")
    # print(f"Average CV Test RMSE: {-scores['test_neg_root_mean_squared_error'].mean():.4f}")
    print(f"Performance Test Set RMSE Liege: {test_rmse_Liege:.4f}")
    print(f"Performance Test Set RMSE KI: {test_rmse_KI:.4f}")
    print(f"Average CV Test R2: nan")
    # print(f"Average CV Test R2: {scores['test_r2'].mean():.4f}")
    print(f"Performance Test Set R2 Liege: {test_r2_Liege:.4f}")
    print(f"Performance Test Set R2 KI: {test_r2_KI:.4f}")
    print(f"Average CV Test Pearson r: nan")
    # print(f"Average CV Test Pearson r: {scores['test_r_corr'].mean():.4f}")
    print(f"Performance Test Set Pearson r Liege: {test_corr_Liege:.4f}")
    print(f"Performance Test Set Pearson r KI: {test_corr_KI:.4f}")
    print(f"Average CV Test Spearman r: nan")
    # print(f"Average CV Test Spearman r: {scores['test_spearmanr'].mean():.4f}")
    print(f"Performance Test Set Spearman r Liege: {test_corr_spearman_Liege:.4f}")
    print(f"Performance Test Set Spearman r KI: {test_corr_spearman_KI:.4f}")
    print("======================================\n")

    # create a dataframe to store the results
    results_df = pd.DataFrame({
        'Model': ['AutoGluon'],
        'Feature Combination': [feature_comb],
        'Target': [target],
        'Average CV Test MAE': 'nan',
        # 'Average CV Test MAE': [-scores['test_neg_mean_absolute_error'].mean()],
        'Performance Test Set MAE Liege': [test_mae_Liege],
        'Performance Test Set MAE KI': [test_mae_KI],
        'Average CV Test RMSE': 'nan',
        # 'Average CV Test RMSE': [-scores['test_neg_root_mean_squared_error'].mean()],
        'Performance Test Set RMSE Liege': [test_rmse_Liege],
        'Performance Test Set RMSE KI': [test_rmse_KI],
        'Average CV Test R2': 'nan',
        # 'Average CV Test R2': [scores['test_r2'].mean()],
        'Performance Test Set R2 Liege': [test_r2_Liege],
        'Performance Test Set R2 KI': [test_r2_KI],
        'Average CV Test Pearson r': 'nan',
        # 'Average CV Test Pearson r': [scores['test_r_corr'].mean()],
        'Performance Test Set Pearson r Liege': [test_corr_Liege],
        'Performance Test Set Pearson r KI': [test_corr_KI],
        'Average CV Test Spearman r': 'nan',
        # 'Average CV Test Spearman r': [scores['test_spearmanr'].mean()],
        'Performance Test Set Spearman r Liege': [test_corr_spearman_Liege],
        'Performance Test Set Spearman r KI': [test_corr_spearman_KI]
    })

    results_df.to_csv(case_results_path + 'results.csv')

else:
    print("\nModel performance:")
    print("======================================")
    print(f"Average CV Test MAE: nan:")
    # print(f"Average CV Test MAE: {-scores['test_neg_mean_absolute_error'].mean():.4f}")
    print(f"Performance Test Set MAE Liege: {test_mae_Liege:.4f}")
    print(f"Average CV Test RMSE: nan")
    # print(f"Average CV Test RMSE: {-scores['test_neg_root_mean_squared_error'].mean():.4f}")
    print(f"Performance Test Set RMSE Liege: {test_rmse_Liege:.4f}")
    print(f"Average CV Test R2: nan")
    # print(f"Average CV Test R2: {scores['test_r2'].mean():.4f}")
    print(f"Performance Test Set R2 Liege: {test_r2_Liege:.4f}")
    print(f"Average CV Test Pearson r: nan")
    # print(f"Average CV Test Pearson r: {scores['test_r_corr'].mean():.4f}")
    print(f"Performance Test Set Pearson r Liege: {test_corr_Liege:.4f}")
    print(f"Average CV Test Spearman r: nan")
    # print(f"Average CV Test Spearman r: {scores['test_spearmanr'].mean():.4f}")
    print(f"Performance Test Set Spearman r Liege: {test_corr_spearman_Liege:.4f}")
    print("======================================\n")

    # create a dataframe to store the results
    results_df = pd.DataFrame({
        'Model': ['AutoGluon'],
        'Feature Combination': [feature_comb],
        'Target': [target],
        'Average CV Test MAE': 'nan',
        # 'Average CV Test MAE': [-scores['test_neg_mean_absolute_error'].mean()],
        'Performance Test Set MAE Liege': [test_mae_Liege],
        'Average CV Test RMSE': 'nan',
        # 'Average CV Test RMSE': [-scores['test_neg_root_mean_squared_error'].mean()],
        'Performance Test Set RMSE Liege': [test_rmse_Liege],
        'Average CV Test R2': 'nan',
        # 'Average CV Test R2': [scores['test_r2'].mean()],
        'Performance Test Set R2 Liege': [test_r2_Liege],
        'Average CV Test Pearson r': 'nan',
        # 'Average CV Test Pearson r': [scores['test_r_corr'].mean()],
        'Performance Test Set Pearson r Liege': [test_corr_Liege],
        'Average CV Test Spearman r': 'nan',
        # 'Average CV Test Spearman r': [scores['test_spearmanr'].mean()],
        'Performance Test Set Spearman r Liege': [test_corr_spearman_Liege],
    })

    results_df.to_csv(case_results_path + 'results.csv')

# %%
print(f"\nDone! AutoGluon pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
