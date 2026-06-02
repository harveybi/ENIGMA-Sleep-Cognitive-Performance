import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import ml_pipeline
from ml_pipeline import MLPipeline

import argparse
import pandas as pd

from joblib import dump

# %%
# reload utils
import importlib
importlib.reload(utils)
importlib.reload(ml_pipeline)
import ml_pipeline
from ml_pipeline import MLPipeline

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='SVM-linear, SHIP_Trend dataset.')
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

# Add num_cores as an optional argument with a default value
parser.add_argument('num_cores', type=int, default=1,
                    help='Number of cores to be used for parallel processing. Default is 1.')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target
num_cores = args.num_cores

print(f"\nStarting SVM-linear pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/SVM-linear/{target}/SHIP_Trend/'

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
print(f"\nStart model training")
case_results_path = results_path + f"{feature_comb}/"
if not os.path.exists(case_results_path):
    os.makedirs(case_results_path)

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml_cld_out
# df_test = df_Liege_ml_cld_out
save_path = case_results_path

ml_pipe_instance = MLPipeline(X, y, df_train, num_cores, save_path)
scores, model = ml_pipe_instance.svm_linear_pipe(stratified_label='Group_Age_SEX')

print(f"Model training completed.\n")

# %%
# save scores and model
scores.to_csv(case_results_path + 'scores.csv')

dump(model, case_results_path + 'model.joblib')

# %%
df_test_Liege = df_Liege_ml_cld_out
y_test_Liege = df_test_Liege[y].values.reshape(-1)
test_mae_Liege, test_rmse_Liege, test_r2_Liege, test_corr_Liege, test_corr_spearman_Liege = ml_pipe_instance.test_perform_plot(df_test_Liege, y_test_Liege, 'Test Set')

df_test_Liege_COF = df_Liege_COF_ml_cld_out
y_test_Liege_COF = df_test_Liege_COF[y].values.reshape(-1)
test_mae_Liege_COF, test_rmse_Liege_COF, test_r2_Liege_COF, test_corr_Liege_COF, test_corr_spearman_Liege_COF = ml_pipe_instance.test_perform_plot(df_test_Liege_COF, y_test_Liege_COF, 'Test Set COF')

df_test_Liege_COGNAP = df_Liege_COGNAP_ml_cld_out
y_test_Liege_COGNAP = df_test_Liege_COGNAP[y].values.reshape(-1)
test_mae_Liege_COGNAP, test_rmse_Liege_COGNAP, test_r2_Liege_COGNAP, test_corr_Liege_COGNAP, test_corr_spearman_Liege_COGNAP = ml_pipe_instance.test_perform_plot(df_test_Liege_COGNAP, y_test_Liege_COGNAP, 'Test Set COGNAP')

df_test_Liege_COF_COGNAP = pd.concat([df_test_Liege_COF, df_test_Liege_COGNAP], ignore_index=True)
y_test_Liege_COF_COGNAP = df_test_Liege_COF_COGNAP[y].values.reshape(-1)
test_mae_Liege_COF_COGNAP, test_rmse_Liege_COF_COGNAP, test_r2_Liege_COF_COGNAP, test_corr_Liege_COF_COGNAP, test_corr_spearman_Liege_COF_COGNAP = ml_pipe_instance.test_perform_plot(df_test_Liege_COF_COGNAP, y_test_Liege_COF_COGNAP, 'Test Set COF_COGNAP')

print("\nModel performance:")
print("======================================")
print(f"Average CV Train MAE: {-scores['train_neg_mean_absolute_error'].mean():.4f}")
print(f"Average CV Test MAE: {-scores['test_neg_mean_absolute_error'].mean():.4f}")
print(f"Performance Test Set MAE Liege: {test_mae_Liege:.4f}")
print(f"Performance Test Set MAE Liege COF: {test_mae_Liege_COF:.4f}")
print(f"Performance Test Set MAE Liege COGNAP: {test_mae_Liege_COGNAP:.4f}")
print(f"Performance Test Set MAE Liege COF_COGNAP: {test_mae_Liege_COF_COGNAP:.4f}")
print(f"Average CV Train RMSE: {-scores['train_neg_root_mean_squared_error'].mean():.4f}")
print(f"Average CV Test RMSE: {-scores['test_neg_root_mean_squared_error'].mean():.4f}")
print(f"Performance Test Set RMSE Liege: {test_rmse_Liege:.4f}")
print(f"Performance Test Set RMSE Liege COF: {test_rmse_Liege_COF:.4f}")
print(f"Performance Test Set RMSE Liege COGNAP: {test_rmse_Liege_COGNAP:.4f}")
print(f"Performance Test Set RMSE Liege COF_COGNAP: {test_rmse_Liege_COF_COGNAP:.4f}")
print(f"Average CV Train R2: {scores['train_r2'].mean():.4f}")
print(f"Average CV Test R2: {scores['test_r2'].mean():.4f}")
print(f"Performance Test Set R2 Liege: {test_r2_Liege:.4f}")
print(f"Performance Test Set R2 Liege COF: {test_r2_Liege_COF:.4f}")
print(f"Performance Test Set R2 Liege COGNAP: {test_r2_Liege_COGNAP:.4f}")
print(f"Performance Test Set R2 Liege COF_COGNAP: {test_r2_Liege_COF_COGNAP:.4f}")
print(f"Average CV Train Pearson r: {scores['train_r_corr'].mean():.4f}")
print(f"Average CV Test Pearson r: {scores['test_r_corr'].mean():.4f}")
print(f"Performance Test Set Pearson r Liege: {test_corr_Liege:.4f}")
print(f"Performance Test Set Pearson r Liege COF: {test_corr_Liege_COF:.4f}")
print(f"Performance Test Set Pearson r Liege COGNAP: {test_corr_Liege_COGNAP:.4f}")
print(f"Performance Test Set Pearson r Liege COF_COGNAP: {test_corr_Liege_COF_COGNAP:.4f}")
print(f"Average CV Train Spearman r: {scores['train_spearmanr'].mean():.4f}")
print(f"Average CV Test Spearman r: {scores['test_spearmanr'].mean():.4f}")
print(f"Performance Test Set Spearman r Liege: {test_corr_spearman_Liege:.4f}")
print(f"Performance Test Set Spearman r Liege COF: {test_corr_spearman_Liege_COF:.4f}")
print(f"Performance Test Set Spearman r Liege COGNAP: {test_corr_spearman_Liege_COGNAP:.4f}")
print(f"Performance Test Set Spearman r Liege COF_COGNAP: {test_corr_spearman_Liege_COF_COGNAP:.4f}")
print("======================================\n")

# create a dataframe to store the results
results_df = pd.DataFrame({
    'Model': ['SVM-linear'],
    'Feature Combination': [feature_comb],
    'Target': [target],
    'Average CV Train MAE': [-scores['train_neg_mean_absolute_error'].mean()],
    'Average CV Test MAE': [-scores['test_neg_mean_absolute_error'].mean()],
    'Performance Test Set MAE Liege': [test_mae_Liege],
    'Performance Test Set MAE Liege COF': [test_mae_Liege_COF],
    'Performance Test Set MAE Liege COGNAP': [test_mae_Liege_COGNAP],
    'Performance Test Set MAE Liege COF_COGNAP': [test_mae_Liege_COF_COGNAP],
    'Average CV Train RMSE': [-scores['train_neg_root_mean_squared_error'].mean()],
    'Average CV Test RMSE': [-scores['test_neg_root_mean_squared_error'].mean()],
    'Performance Test Set RMSE Liege': [test_rmse_Liege],
    'Performance Test Set RMSE Liege COF': [test_rmse_Liege_COF],
    'Performance Test Set RMSE Liege COGNAP': [test_rmse_Liege_COGNAP],
    'Performance Test Set RMSE Liege COF_COGNAP': [test_rmse_Liege_COF_COGNAP],
    'Average CV Train R2': [scores['train_r2'].mean()],
    'Average CV Test R2': [scores['test_r2'].mean()],
    'Performance Test Set R2 Liege': [test_r2_Liege],
    'Performance Test Set R2 Liege COF': [test_r2_Liege_COF],
    'Performance Test Set R2 Liege COGNAP': [test_r2_Liege_COGNAP],
    'Performance Test Set R2 Liege COF_COGNAP': [test_r2_Liege_COF_COGNAP],
    'Average CV Train Pearson r': [scores['train_r_corr'].mean()],
    'Average CV Test Pearson r': [scores['test_r_corr'].mean()],
    'Performance Test Set Pearson r Liege': [test_corr_Liege],
    'Performance Test Set Pearson r Liege COF': [test_corr_Liege_COF],
    'Performance Test Set Pearson r Liege COGNAP': [test_corr_Liege_COGNAP],
    'Performance Test Set Pearson r Liege COF_COGNAP': [test_corr_Liege_COF_COGNAP],
    'Average CV Train Spearman r': [scores['train_spearmanr'].mean()],
    'Average CV Test Spearman r': [scores['test_spearmanr'].mean()],
    'Performance Test Set Spearman r Liege': [test_corr_spearman_Liege],
    'Performance Test Set Spearman r Liege COF': [test_corr_spearman_Liege_COF],
    'Performance Test Set Spearman r Liege COGNAP': [test_corr_spearman_Liege_COGNAP],
    'Performance Test Set Spearman r Liege COF_COGNAP': [test_corr_spearman_Liege_COGNAP]
})

results_df.to_csv(case_results_path + 'results.csv')

# %%
print(f"\nDone! SVM-linear pipeline for {target} prediction by {feature_comb} completed successfully.")
# print Date and Time
from datetime import datetime
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
