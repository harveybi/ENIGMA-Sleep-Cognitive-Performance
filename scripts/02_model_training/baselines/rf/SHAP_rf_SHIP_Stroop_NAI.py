import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import ml_pipeline
from ml_pipeline import MLPipeline

import argparse
import pandas as pd
import shap
import pickle
from joblib import dump, load
from datetime import datetime

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='SHAP for Random Forest, SHIP_Trend dataset.')
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
''')

parser.add_argument('target', type=str, help='''Name of the target to be predicted.
    Available targets are:
    - Stroop: Stroop_Test
    - Memory: NAI_Wordlist_Test
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target

print(f"\nStarting SHAP for Random Forest pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/rf/{target}/SHIP_Trend/'

# %%
df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed.csv')
df_Liege_COF = pd.read_csv(data_save_path + 'Liege_COF_dataset_renamed.csv')
df_Liege_COGNAP = pd.read_csv(data_save_path + 'Liege_COGNAP_dataset_renamed.csv')

# %%
"""
Define feature lists
"""
Sleep = ['PSG_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Dur', 'Self_Sleep_Eff', 'Depression_score']
Cov = ['Age_at_Scan', 'SEX', 'BMI']
TIV = 'EstimatedTotalIntraCranialVol'

columns = df_SHIP.columns.tolist()
Thickness_DK = columns[columns.index('lh_bankssts_thickness'):columns.index('rh_insula_thickness')+1]
Thickness_Schaefer = columns[columns.index('LH_Vis_1_thickness'):columns.index('RH_Default_pCunPCC_9_thickness')+1]
Area_DK = columns[columns.index('lh_bankssts_area'):columns.index('rh_insula_area')+1]
Area_Schaefer = columns[columns.index('LH_Vis_1_area'):columns.index('RH_Default_pCunPCC_9_area')+1]
Subcortical = columns[columns.index('Left-Lateral-Ventricle'):columns.index('CC_Anterior')+1]

targets = ['Stroop_Test', 'Memory_Test']

# %%
"""
Data preprocessing for SHIP_Trend dataset
1. Convert sleep measurements units
2. Remove subjects with missing data in necessary features
3. Outlier detection and removal
4. Add group based on age and sex
5. Brain correction by brain size using internal data normalisation
6. NAI_Wordlist_Test transfer to accuracy
7. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']
df_SHIP_ml = utils.convert_units(df_SHIP, sleep_dur_cols, sleep_eff_cols)

# remove subjects with missing data in necessary features
important_variables_list = Sleep + Cov + [TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + targets
df_SHIP_ml_cld = utils.clean_missing_data(df_SHIP_ml, important_variables_list)

# outlier detection and removal
df_SHIP_ml_cld_out = utils.rm_outliers(df_SHIP_ml_cld, important_variables_list)
# reset index
df_SHIP_ml_cld_out.reset_index(drop=True, inplace=True)

# add a column of 'Age_Group' after 'Age_at_Scan'.
df_SHIP_ml_cld_out = utils.add_age_groups(df_SHIP_ml_cld_out)

# add a column of Group which represents the group of age groups and SEX groups
df_SHIP_ml_cld_out = utils.add_groups_age_sex(df_SHIP_ml_cld_out)

print("Data type of SEX column before conversion:", df_SHIP_ml_cld_out['SEX'].dtype)
# Convert to int if it's not already an int
if df_SHIP_ml_cld_out['SEX'].dtype != 'int64':
    df_SHIP_ml_cld_out['SEX'] = df_SHIP_ml_cld_out['SEX'].astype(int)
print("Data type of SEX column after conversion:", df_SHIP_ml_cld_out['SEX'].dtype)

# %%
# Brain correction by brain size using internal data normalisation
df_SHIP_ml_cld_out[Thickness_DK] = df_SHIP_ml_cld_out[Thickness_DK].div(df_SHIP_ml_cld_out[Thickness_DK].sum(axis=1), axis=0)
df_SHIP_ml_cld_out[Thickness_Schaefer] = df_SHIP_ml_cld_out[Thickness_Schaefer].div(df_SHIP_ml_cld_out[Thickness_Schaefer].sum(axis=1), axis=0)
df_SHIP_ml_cld_out[Area_DK] = df_SHIP_ml_cld_out[Area_DK].div(df_SHIP_ml_cld_out[Area_DK].sum(axis=1), axis=0)
df_SHIP_ml_cld_out[Area_Schaefer] = df_SHIP_ml_cld_out[Area_Schaefer].div(df_SHIP_ml_cld_out[Area_Schaefer].sum(axis=1), axis=0)
df_SHIP_ml_cld_out[Subcortical] = df_SHIP_ml_cld_out[Subcortical].div(df_SHIP_ml_cld_out['EstimatedTotalIntraCranialVol'], axis=0)

# NAI_Wordlist_Test transfer to accuracy
df_SHIP_ml_cld_out['Memory_Test'] = df_SHIP_ml_cld_out['Memory_Test'].apply(lambda x: x/16) * 100

# %%
# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_SHIP_ml_cld_out[Sleep] = df_SHIP_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_SHIP_ml_cld_out[Subcortical] = df_SHIP_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_SHIP_ml_cld_out[Sleep] = df_SHIP_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_SHIP_ml_cld_out[Sleep] = df_SHIP_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_SHIP_ml_cld_out[Thickness_DK] = df_SHIP_ml_cld_out[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml_cld_out[Thickness_Schaefer] = df_SHIP_ml_cld_out[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml_cld_out[Area_DK] = df_SHIP_ml_cld_out[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml_cld_out[Area_Schaefer] = df_SHIP_ml_cld_out[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml_cld_out[Subcortical] = df_SHIP_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_SHIP_ml_cld_out[Subcortical] = df_SHIP_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Data preprocessing for Liege dataset
1. Convert sleep measurements units
2. Remove subjects with missing data in necessary features
3. Outlier detection and removal
4. Brain correction by brain size using internal data normalisation
5. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Liege_ml = utils.convert_units(df_Liege, sleep_dur_cols, sleep_eff_cols)

# remove subjects with missing data in necessary features
df_Liege_ml_cld = utils.clean_missing_data(df_Liege_ml, important_variables_list)

# outlier detection and removal
df_Liege_ml_cld_out = utils.rm_outliers(df_Liege_ml_cld, important_variables_list)
# reset index
df_Liege_ml_cld_out.reset_index(drop=True, inplace=True)

# Brain correction by brain size using internal data normalisation
df_Liege_ml_cld_out[Thickness_DK] = df_Liege_ml_cld_out[Thickness_DK].div(df_Liege_ml_cld_out[Thickness_DK].sum(axis=1), axis=0)
df_Liege_ml_cld_out[Thickness_Schaefer] = df_Liege_ml_cld_out[Thickness_Schaefer].div(df_Liege_ml_cld_out[Thickness_Schaefer].sum(axis=1), axis=0)
df_Liege_ml_cld_out[Area_DK] = df_Liege_ml_cld_out[Area_DK].div(df_Liege_ml_cld_out[Area_DK].sum(axis=1), axis=0)
df_Liege_ml_cld_out[Area_Schaefer] = df_Liege_ml_cld_out[Area_Schaefer].div(df_Liege_ml_cld_out[Area_Schaefer].sum(axis=1), axis=0)
df_Liege_ml_cld_out[Subcortical] = df_Liege_ml_cld_out[Subcortical].div(df_Liege_ml_cld_out['EstimatedTotalIntraCranialVol'], axis=0)

df_Liege_ml_cld_out['Memory_Test'] = df_Liege_ml_cld_out['Memory_Test'] * 100
df_Liege_ml_cld_out['1-back_acc'] = df_Liege_ml_cld_out['1-back_acc'] * 100
df_Liege_ml_cld_out['2-back_acc'] = df_Liege_ml_cld_out['2-back_acc'] * 100
df_Liege_ml_cld_out['3-back_acc'] = df_Liege_ml_cld_out['3-back_acc'] * 100

# %%
# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Liege_ml_cld_out[Sleep] = df_Liege_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Liege_ml_cld_out[Subcortical] = df_Liege_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_Liege_ml_cld_out[Sleep] = df_Liege_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_Liege_ml_cld_out[Sleep] = df_Liege_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_ml_cld_out[Thickness_DK] = df_Liege_ml_cld_out[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml_cld_out[Thickness_Schaefer] = df_Liege_ml_cld_out[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml_cld_out[Area_DK] = df_Liege_ml_cld_out[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml_cld_out[Area_Schaefer] = df_Liege_ml_cld_out[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml_cld_out[Subcortical] = df_Liege_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_ml_cld_out[Subcortical] = df_Liege_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Data preprocessing for Liege COF dataset
1. Convert sleep measurements units
2. Remove subjects with missing data in necessary features
3. Outlier detection and removal
4. Brain correction by brain size using internal data normalisation
5. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Liege_COF_ml = utils.convert_units(df_Liege_COF, sleep_dur_cols, sleep_eff_cols)

# remove subjects with missing data in necessary features
df_Liege_COF_ml_cld = utils.clean_missing_data(df_Liege_COF_ml, important_variables_list)

# outlier detection and removal
df_Liege_COF_ml_cld_out = utils.rm_outliers(df_Liege_COF_ml_cld, important_variables_list)
# reset index
df_Liege_COF_ml_cld_out.reset_index(drop=True, inplace=True)

# Brain correction by brain size using internal data normalisation
df_Liege_COF_ml_cld_out[Thickness_DK] = df_Liege_COF_ml_cld_out[Thickness_DK].div(df_Liege_COF_ml_cld_out[Thickness_DK].sum(axis=1), axis=0)
df_Liege_COF_ml_cld_out[Thickness_Schaefer] = df_Liege_COF_ml_cld_out[Thickness_Schaefer].div(df_Liege_COF_ml_cld_out[Thickness_Schaefer].sum(axis=1), axis=0)
df_Liege_COF_ml_cld_out[Area_DK] = df_Liege_COF_ml_cld_out[Area_DK].div(df_Liege_COF_ml_cld_out[Area_DK].sum(axis=1), axis=0)
df_Liege_COF_ml_cld_out[Area_Schaefer] = df_Liege_COF_ml_cld_out[Area_Schaefer].div(df_Liege_COF_ml_cld_out[Area_Schaefer].sum(axis=1), axis=0)
df_Liege_COF_ml_cld_out[Subcortical] = df_Liege_COF_ml_cld_out[Subcortical].div(df_Liege_COF_ml_cld_out['EstimatedTotalIntraCranialVol'], axis=0)

df_Liege_COF_ml_cld_out['Memory_Test'] = df_Liege_COF_ml_cld_out['Memory_Test'] * 100
df_Liege_COF_ml_cld_out['1-back_acc'] = df_Liege_COF_ml_cld_out['1-back_acc'] * 100
df_Liege_COF_ml_cld_out['2-back_acc'] = df_Liege_COF_ml_cld_out['2-back_acc'] * 100
df_Liege_COF_ml_cld_out['3-back_acc'] = df_Liege_COF_ml_cld_out['3-back_acc'] * 100

# %%
# shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Liege_COF_ml_cld_out[Sleep] = df_Liege_COF_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Liege_COF_ml_cld_out[Subcortical] = df_Liege_COF_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_Liege_COF_ml_cld_out[Sleep] = df_Liege_COF_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_Liege_COF_ml_cld_out[Sleep] = df_Liege_COF_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_COF_ml_cld_out[Thickness_DK] = df_Liege_COF_ml_cld_out[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml_cld_out[Thickness_Schaefer] = df_Liege_COF_ml_cld_out[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml_cld_out[Area_DK] = df_Liege_COF_ml_cld_out[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml_cld_out[Area_Schaefer] = df_Liege_COF_ml_cld_out[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml_cld_out[Subcortical] = df_Liege_COF_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_COF_ml_cld_out[Subcortical] = df_Liege_COF_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Data preprocessing for Liege COGNAP dataset
1. Convert sleep measurements units
2. Remove subjects with missing data in necessary features
3. Outlier detection and removal
4. Brain correction by brain size using internal data normalisation
5. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Liege_COGNAP_ml = utils.convert_units(df_Liege_COGNAP, sleep_dur_cols, sleep_eff_cols)

# remove subjects with missing data in necessary features
df_Liege_COGNAP_ml_cld = utils.clean_missing_data(df_Liege_COGNAP_ml, important_variables_list)

# outlier detection and removal
df_Liege_COGNAP_ml_cld_out = utils.rm_outliers(df_Liege_COGNAP_ml_cld, important_variables_list)
# reset index
df_Liege_COGNAP_ml_cld_out.reset_index(drop=True, inplace=True)

# Brain correction by brain size using internal data normalisation
df_Liege_COGNAP_ml_cld_out[Thickness_DK] = df_Liege_COGNAP_ml_cld_out[Thickness_DK].div(df_Liege_COGNAP_ml_cld_out[Thickness_DK].sum(axis=1), axis=0)
df_Liege_COGNAP_ml_cld_out[Thickness_Schaefer] = df_Liege_COGNAP_ml_cld_out[Thickness_Schaefer].div(df_Liege_COGNAP_ml_cld_out[Thickness_Schaefer].sum(axis=1), axis=0)
df_Liege_COGNAP_ml_cld_out[Area_DK] = df_Liege_COGNAP_ml_cld_out[Area_DK].div(df_Liege_COGNAP_ml_cld_out[Area_DK].sum(axis=1), axis=0)
df_Liege_COGNAP_ml_cld_out[Area_Schaefer] = df_Liege_COGNAP_ml_cld_out[Area_Schaefer].div(df_Liege_COGNAP_ml_cld_out[Area_Schaefer].sum(axis=1), axis=0)
df_Liege_COGNAP_ml_cld_out[Subcortical] = df_Liege_COGNAP_ml_cld_out[Subcortical].div(df_Liege_COGNAP_ml_cld_out['EstimatedTotalIntraCranialVol'], axis=0)

df_Liege_COGNAP_ml_cld_out['Memory_Test'] = df_Liege_COGNAP_ml_cld_out['Memory_Test'] * 100
df_Liege_COGNAP_ml_cld_out['1-back_acc'] = df_Liege_COGNAP_ml_cld_out['1-back_acc'] * 100
df_Liege_COGNAP_ml_cld_out['2-back_acc'] = df_Liege_COGNAP_ml_cld_out['2-back_acc'] * 100
df_Liege_COGNAP_ml_cld_out['3-back_acc'] = df_Liege_COGNAP_ml_cld_out['3-back_acc'] * 100

# %%
# shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Liege_COGNAP_ml_cld_out[Sleep] = df_Liege_COGNAP_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Liege_COGNAP_ml_cld_out[Subcortical] = df_Liege_COGNAP_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_Liege_COGNAP_ml_cld_out[Sleep] = df_Liege_COGNAP_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_Liege_COGNAP_ml_cld_out[Sleep] = df_Liege_COGNAP_ml_cld_out[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_COGNAP_ml_cld_out[Thickness_DK] = df_Liege_COGNAP_ml_cld_out[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml_cld_out[Thickness_Schaefer] = df_Liege_COGNAP_ml_cld_out[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml_cld_out[Area_DK] = df_Liege_COGNAP_ml_cld_out[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml_cld_out[Area_Schaefer] = df_Liege_COGNAP_ml_cld_out[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml_cld_out[Subcortical] = df_Liege_COGNAP_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_COGNAP_ml_cld_out[Subcortical] = df_Liege_COGNAP_ml_cld_out[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Define X and y
"""
X_dict = {
    'Sleep': Sleep,
    'Cov': Cov,
    'Brain': Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'CT': Thickness_DK + Thickness_Schaefer,
    'SA': Area_DK + Area_Schaefer,
    'Subcor': Subcortical,
    'Sleep_Cov': Sleep + Cov,
    'Sleep_Cov_Brain': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_Cov_CT': Sleep + Cov + Thickness_DK + Thickness_Schaefer,
    'Sleep_Cov_SA': Sleep + Cov + Area_DK + Area_Schaefer,
    'Sleep_Cov_Subcor': Sleep + Cov + Subcortical,
    'Sleep_Brain': Sleep + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_CT': Sleep + Thickness_DK + Thickness_Schaefer,
    'Sleep_SA': Sleep + Area_DK + Area_Schaefer,
    'Sleep_Subcor': Sleep + Subcortical,
    'Cov_Brain': Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Cov_CT': Cov + Thickness_DK + Thickness_Schaefer,
    'Cov_SA': Cov + Area_DK + Area_Schaefer,
    'Cov_Subcor': Cov + Subcortical,
    'Sleep_Shuffle_Cov': Sleep + Cov,
    'Sleep_Cov_Subcor_Shuffle': Sleep + Cov + Subcortical,
    'Sleep_Shuffle_Cov_Subcor': Sleep + Cov + Subcortical,
    'Sleep_Shuffle_Subcor': Sleep + Subcortical,
    'Cov_Brain_Shuffle': Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Cov_Subcor_Shuffle': Cov + Subcortical
}

y_dict = {
    'Stroop': 'Stroop_Test',
    'Memory': 'Memory_Test'
}

# %%
'''
SHAP explanation
'''
print("\nStart SHAP explanation")

case_results_path = results_path + f"{feature_comb}/"

SHAP_path = case_results_path + 'SHAP/'
if not os.path.exists(SHAP_path):
    os.makedirs(SHAP_path)

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml

# load the model by joblib
model = load(case_results_path + 'model.joblib')

explainer_train_set = shap.explainers.Permutation(model.predict, df_SHIP_ml_cld_out[X], max_evals=2000)

# %%
# explanation for the training set
explanation_train_set_train = explainer_train_set(df_SHIP_ml_cld_out[X])
with open(SHAP_path + 'explainer_train_set_train.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_train, f)
# explanation for the test set
explanation_train_set_test_Liege = explainer_train_set(df_Liege_ml_cld_out[X])
with open(SHAP_path + 'explainer_train_set_test_Liege.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_test_Liege, f)

explanation_train_set_test_Liege_COF = explainer_train_set(df_Liege_COF_ml_cld_out[X])
with open(SHAP_path + 'explainer_train_set_test_Liege_COF.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_test_Liege_COF, f)

explanation_train_set_test_Liege_COGNAP = explainer_train_set(df_Liege_COGNAP_ml_cld_out[X])
with open(SHAP_path + 'explainer_train_set_test_Liege_COGNAP.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_test_Liege_COGNAP, f)

# %%
ml_pipe_instance = MLPipeline(X, y)

ml_pipe_instance.shap_explain(explanation_train_set_train, SHAP_path, 'train_set_train')
ml_pipe_instance.shap_explain(explanation_train_set_test_Liege, SHAP_path, 'train_set_test_Liege')
ml_pipe_instance.shap_explain(explanation_train_set_test_Liege_COF, SHAP_path, 'train_set_test_Liege_COF')
ml_pipe_instance.shap_explain(explanation_train_set_test_Liege_COGNAP, SHAP_path, 'train_set_test_Liege_COGNAP')

print("SHAP explanation completed.\n")

# %%
print(f"\nDone! SHAP Random Forest pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))