import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import advanced_ml_pipeline
from advanced_ml_pipeline import AdMLPipeline

import argparse
import pandas as pd
import shap
import pickle
from datetime import datetime
from autogluon.tabular import TabularPredictor

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
    - Sleep_APOE: Sleep + APOE
    - Sleep_APOE_Shuffle: Sleep + APOE (Shuffled)
    - Sleep_Cov_APOE: Sleep + Cov + APOE
    - Sleep_Cov_APOE_Shuffle: Sleep + Cov + APOE (Shuffled)
    - Cov_APOE: Cov + APOE
    - Cov_APOE_Shuffle: Cov + APOE (Shuffled)
    - Brain_APOE: Brain + APOE
    - Brain_APOE_Shuffle: Brain + APOE (Shuffled)
    - Subcor_APOE: Subcor + APOE
    - Subcor_APOE_Shuffle: Subcor + APOE (Shuffled)
    - Sleep_Cov_Brain_APOE: Sleep + Cov + Brain + APOE
    - Sleep_Cov_Brain_APOE_Shuffle: Sleep + Cov + Brain + APOE (Shuffled)
    - Sleep_Cov_Subcor_APOE: Sleep + Cov + Subcor + APOE
    - Sleep_Cov_Subcor_APOE_Shuffle: Sleep + Cov + Subcor + APOE (Shuffled)
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

print(f"\nStarting AutoGluon pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
model_save_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/JURECA_AutoGluon/{target}/SHIP_Trend/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/CV_AutoGluon/{target}/SHIP_Trend/'

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
if 'APOE_Shuffle' in feature_comb:
    df_Liege_ml[APOE4] = df_Liege_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Data preprocessing for Liege COF dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Liege_COF_ml = utils.convert_units(df_Liege_COF, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_Liege_COF_ml[Thickness_DK] = df_Liege_COF_ml[Thickness_DK].div(df_Liege_COF_ml[Thickness_DK].sum(axis=1), axis=0)
df_Liege_COF_ml[Thickness_Schaefer] = df_Liege_COF_ml[Thickness_Schaefer].div(df_Liege_COF_ml[Thickness_Schaefer].sum(axis=1), axis=0)
df_Liege_COF_ml[Area_DK] = df_Liege_COF_ml[Area_DK].div(df_Liege_COF_ml[Area_DK].sum(axis=1), axis=0)
df_Liege_COF_ml[Area_Schaefer] = df_Liege_COF_ml[Area_Schaefer].div(df_Liege_COF_ml[Area_Schaefer].sum(axis=1), axis=0)
df_Liege_COF_ml[Subcortical] = df_Liege_COF_ml[Subcortical].div(df_Liege_COF_ml['EstimatedTotalIntraCranialVol'], axis=0)

df_Liege_COF_ml['Memory_Test'] = df_Liege_COF_ml['Memory_Test'] * 100
df_Liege_COF_ml['1-back_acc'] = df_Liege_COF_ml['1-back_acc'] * 100
df_Liege_COF_ml['2-back_acc'] = df_Liege_COF_ml['2-back_acc'] * 100
df_Liege_COF_ml['3-back_acc'] = df_Liege_COF_ml['3-back_acc'] * 100

# shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Liege_COF_ml[Sleep] = df_Liege_COF_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Liege_COF_ml[Subcortical] = df_Liege_COF_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_Liege_COF_ml[Sleep] = df_Liege_COF_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_Liege_COF_ml[Sleep] = df_Liege_COF_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_COF_ml[Thickness_DK] = df_Liege_COF_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Thickness_Schaefer] = df_Liege_COF_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Area_DK] = df_Liege_COF_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Area_Schaefer] = df_Liege_COF_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Subcortical] = df_Liege_COF_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_COF_ml[Subcortical] = df_Liege_COF_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_Liege_COF_ml[APOE4] = df_Liege_COF_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Data preprocessing for Liege COGNAP dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Liege_COGNAP_ml = utils.convert_units(df_Liege_COGNAP, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_Liege_COGNAP_ml[Thickness_DK] = df_Liege_COGNAP_ml[Thickness_DK].div(df_Liege_COGNAP_ml[Thickness_DK].sum(axis=1), axis=0)
df_Liege_COGNAP_ml[Thickness_Schaefer] = df_Liege_COGNAP_ml[Thickness_Schaefer].div(df_Liege_COGNAP_ml[Thickness_Schaefer].sum(axis=1), axis=0)
df_Liege_COGNAP_ml[Area_DK] = df_Liege_COGNAP_ml[Area_DK].div(df_Liege_COGNAP_ml[Area_DK].sum(axis=1), axis=0)
df_Liege_COGNAP_ml[Area_Schaefer] = df_Liege_COGNAP_ml[Area_Schaefer].div(df_Liege_COGNAP_ml[Area_Schaefer].sum(axis=1), axis=0)
df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].div(df_Liege_COGNAP_ml['EstimatedTotalIntraCranialVol'], axis=0)

df_Liege_COGNAP_ml['Memory_Test'] = df_Liege_COGNAP_ml['Memory_Test'] * 100
df_Liege_COGNAP_ml['1-back_acc'] = df_Liege_COGNAP_ml['1-back_acc'] * 100
df_Liege_COGNAP_ml['2-back_acc'] = df_Liege_COGNAP_ml['2-back_acc'] * 100
df_Liege_COGNAP_ml['3-back_acc'] = df_Liege_COGNAP_ml['3-back_acc'] * 100

# shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Liege_COGNAP_ml[Sleep] = df_Liege_COGNAP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_Liege_COGNAP_ml[Sleep] = df_Liege_COGNAP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_Liege_COGNAP_ml[Sleep] = df_Liege_COGNAP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_COGNAP_ml[Thickness_DK] = df_Liege_COGNAP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml[Thickness_Schaefer] = df_Liege_COGNAP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml[Area_DK] = df_Liege_COGNAP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml[Area_Schaefer] = df_Liege_COGNAP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_Liege_COGNAP_ml[APOE4] = df_Liege_COGNAP_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)

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
if 'APOE_Shuffle' in feature_comb:
    df_KI_ml[APOE4] = df_KI_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)

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
    'Cov_Subcor_Shuffle': Cov + Subcortical,
    'Sleep_APOE': Sleep + APOE4,
    'Sleep_APOE_Shuffle': Sleep + APOE4,
    'Sleep_Cov_APOE': Sleep + Cov + APOE4,
    'Sleep_Cov_APOE_Shuffle': Sleep + Cov + APOE4,
    'Cov_APOE': Cov + APOE4,
    'Cov_APOE_Shuffle': Cov + APOE4,
    'Brain_APOE': Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + APOE4,
    'Brain_APOE_Shuffle': Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + APOE4,
    'Subcor_APOE': Subcortical + APOE4,
    'Subcor_APOE_Shuffle': Subcortical + APOE4,
    'Sleep_Cov_Brain_APOE': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + APOE4,
    'Sleep_Cov_Brain_APOE_Shuffle': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + APOE4,
    'Sleep_Cov_Subcor_APOE': Sleep + Cov + Subcortical + APOE4,
    'Sleep_Cov_Subcor_APOE_Shuffle': Sleep + Cov + Subcortical + APOE4
}

y_dict = {
    'Stroop': 'Stroop_Test',
    'Memory': 'Memory_Test'
}

# %%
# case_results_path = results_path + f"{feature_comb}/"
# if not os.path.exists(case_results_path):
#     os.makedirs(case_results_path)
predictor_save_path = model_save_path + f"{feature_comb}/" + 'autogluon/'

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml

model = TabularPredictor.load(predictor_save_path)

# %%
model.refit_full()

# %%
# explore how to retrain the trained model correctly
leaderboard = model.leaderboard(extra_info=True)

# %%
leaderboard_minimal = leaderboard[['model', 'stack_level', 'model_type', 'child_model_type', 'hyperparameters', 'ag_args_fit', 'child_hyperparameters', 'child_ag_args_fit', 'features']]

# %%
new_hyperparameters = {
    'NeuralNetFastAI_r172_BAG_L1': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'layers': [400], 'emb_drop': 0.05604276533830355, 'ps': 0.022591301744255762, 'bs': 512, 'lr': 0.027320709383189166, 'epochs': 32, 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'NNFastAiTabularModel'},
}

# %%
import json

# Function to clean hyperparameters dictionary
def clean_hyperparameters(hyperparameters):
    cleaned_hyperparameters = {}
    for key, value in hyperparameters.items():
        if isinstance(value, dict):
            cleaned_hyperparameters[key] = clean_hyperparameters(value)
        elif isinstance(value, list):
            cleaned_hyperparameters[key] = [clean_hyperparameters(v) if isinstance(v, dict) else v for v in value]
        else:
            if not isinstance(key, slice):
                cleaned_hyperparameters[key] = value
    return cleaned_hyperparameters

# Initialize the new hyperparameters dictionary
new_hyperparameters = {}

# Loop through each model in the ensemble and extract the hyperparameters
for i, row in leaderboard_minimal.iterrows():
    model_name = row['model']
    model_hyperparameters = row['hyperparameters']
    model_ag_args_fit = row['ag_args_fit']
    model_child_hyperparameters = row['child_hyperparameters']
    model_child_ag_args_fit = row['child_ag_args_fit']
    model_stack_level = row['stack_level']
    model_type = row['model_type']
    model_child_type = row['child_model_type']

    # Flatten and clean hyperparameters
    new_hyperparameters[model_name] = {
        'hyperparameters': clean_hyperparameters(model_hyperparameters),
        'ag_args_fit': clean_hyperparameters(model_ag_args_fit),
        'child_hyperparameters': clean_hyperparameters(model_child_hyperparameters),
        'child_ag_args_fit': clean_hyperparameters(model_child_ag_args_fit),
        'stack_level': model_stack_level,
        'model_type': model_type,
        'child_model_type': model_child_type
    }

# Debugging: Print the new_hyperparameters to identify any issues
print(json.dumps(new_hyperparameters, indent=4))

# Example simple configuration
simple_hyperparameters = {
    'GBM': {
        'learning_rate': 0.1,
        'num_boost_round': 100,
    }
}

# Retrain the model with the validated hyperparameters
test_refit_save_path = results_path + f"{feature_comb}/" + 'autogluon_refit/'

try:
    predictor = TabularPredictor(
        label=y,
        problem_type='regression',
        eval_metric='r2',
        path=test_refit_save_path
    ).fit(
        train_data=df_train[X + [y]],
        hyperparameters=new_hyperparameters,  # Start with a simple configuration: simple_hyperparameters
        presets='best_quality',
        refit_full='best',
        keep_only_best=True,  # Keep only the best model (and its ancestors)
        num_cpus=num_cores,
        save_space=True,
        time_limit=60,  # only for testing
    )
except Exception as e:
    print(f"Error: {e}")

# %%
# {
#     'WeightedEnsemble_L2': {'hyperparameters': {'use_orig_features': False, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'ensemble_size': 25, 'subsample_size': 1000000}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'stack_level': 2, 'model_type': 'WeightedEnsembleModel', 'child_model_type': 'GreedyWeightedEnsembleModel'},
#     'NeuralNetFastAI_r143_BAG_L1': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'layers': [200, 100, 50], 'emb_drop': 0.6239200452002372, 'ps': 0.670815151683455, 'bs': 1024, 'lr': 0.07170321592506483, 'epochs': 39, 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'NNFastAiTabularModel'},
#     'NeuralNetFastAI_r172_BAG_L1': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'layers': [400], 'emb_drop': 0.05604276533830355, 'ps': 0.022591301744255762, 'bs': 512, 'lr': 0.027320709383189166, 'epochs': 32, 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'NNFastAiTabularModel'},
#     'NeuralNetFastAI_r187_BAG_L1': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'layers': [200, 100, 50], 'emb_drop': 0.5074958658302495, 'ps': 0.34814978753283593, 'bs': 1024, 'lr': 0.026342427824862867, 'epochs': 42, 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'NNFastAiTabularModel'},
#     'NeuralNetFastAI_r4_BAG_L1': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'layers': [200], 'emb_drop': 0.06099050979107849, 'ps': 0.5447097256648953, 'bs': 256, 'lr': 0.04119582873110387, 'epochs': 39, 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'NNFastAiTabularModel'},
#     'NeuralNetTorch_r197_BAG_L1': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'num_epochs': 500, 'epochs_wo_improve': 20, 'activation': 'elu', 'embedding_size_factor': 1.0, 'embed_exponent': 0.56, 'max_embedding_dim': 100, 'y_range': None, 'y_range_extend': 0.05, 'dropout_prob': 0.18109219857068798, 'optimizer': 'adam', 'learning_rate': 0.00634181748507711, 'weight_decay': 5.3861175580695396e-08, 'proc.embed_min_categories': 4, 'proc.impute_strategy': 'median', 'proc.max_category_levels': 100, 'proc.skew_threshold': 0.99, 'use_ngram_features': False, 'num_layers': 1, 'hidden_size': 250, 'max_batch_size': 512, 'use_batchnorm': False, 'loss_function': 'auto'}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'TabularNeuralNetTorchModel'},
#     'WeightedEnsemble_L2_FULL': {'hyperparameters': {'use_orig_features': False, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'ensemble_size': 25, 'subsample_size': 1000000}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'stack_level': 2, 'model_type': 'WeightedEnsembleModel', 'child_model_type': 'GreedyWeightedEnsembleModel'}, 'NeuralNetTorch_r197_BAG_L1_FULL': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.15, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'num_epochs': 25, 'epochs_wo_improve': 20, 'activation': 'elu', 'embedding_size_factor': 1.0, 'embed_exponent': 0.56, 'max_embedding_dim': 100, 'y_range': None, 'y_range_extend': 0.05, 'dropout_prob': 0.18109219857068798, 'optimizer': 'adam', 'learning_rate': 0.00634181748507711, 'weight_decay': 5.3861175580695396e-08, 'proc.embed_min_categories': 4, 'proc.impute_strategy': 'median', 'proc.max_category_levels': 100, 'proc.skew_threshold': 0.99, 'use_ngram_features': False, 'num_layers': 1, 'hidden_size': 250, 'max_batch_size': 512, 'use_batchnorm': False, 'loss_function': 'auto', 'batch_size': 32}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'TabularNeuralNetTorchModel'}, 'NeuralNetFastAI_r4_BAG_L1_FULL': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.15, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'layers': [200], 'emb_drop': 0.06099050979107849, 'ps': 0.5447097256648953, 'bs': 256, 'lr': 0.04119582873110387, 'epochs': 39, 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0, 'best_epoch': 17}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'NNFastAiTabularModel'},
#     'NeuralNetFastAI_r187_BAG_L1_FULL': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.15, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'layers': [200, 100, 50], 'emb_drop': 0.5074958658302495, 'ps': 0.34814978753283593, 'bs': 1024, 'lr': 0.026342427824862867, 'epochs': 42, 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0, 'best_epoch': 17}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'NNFastAiTabularModel'}, 'NeuralNetFastAI_r172_BAG_L1_FULL': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.15, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'layers': [400], 'emb_drop': 0.05604276533830355, 'ps': 0.022591301744255762, 'bs': 512, 'lr': 0.027320709383189166, 'epochs': 32, 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0, 'best_epoch': 16}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'NNFastAiTabularModel'}, 'NeuralNetFastAI_r143_BAG_L1_FULL': {'hyperparameters': {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5, 'save_bag_folds': True}, 'ag_args_fit': {'max_memory_usage_ratio': 1.15, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': None, 'valid_special_types': None, 'ignored_type_group_special': None, 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None, 'drop_unique': False}, 'child_hyperparameters': {'layers': [200, 100, 50], 'emb_drop': 0.6239200452002372, 'ps': 0.670815151683455, 'bs': 1024, 'lr': 0.07170321592506483, 'epochs': 39, 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0, 'best_epoch': 12}, 'child_ag_args_fit': {'max_memory_usage_ratio': 1.0, 'max_time_limit_ratio': 1.0, 'max_time_limit': None, 'min_time_limit': 0, 'valid_raw_types': ['bool', 'int', 'float', 'category'], 'valid_special_types': None, 'ignored_type_group_special': ['text_ngram', 'text_as_category'], 'ignored_type_group_raw': None, 'get_features_kwargs': None, 'get_features_kwargs_extra': None, 'predict_1_batch_size': None, 'temperature_scalar': None}, 'stack_level': 1, 'model_type': 'StackerEnsembleModel', 'child_model_type': 'NNFastAiTabularModel'}
# }

# %%
# test_refit_save_path = results_path + f"{feature_comb}/" + 'autogluon_refit/'
# # Retrain the model with the new hyperparameters
# predictor = TabularPredictor(
#             label=y,
#             problem_type='regression',
#             eval_metric='r2',
#             path=test_refit_save_path
#         ).fit(
#             train_data=df_train[X + [y]],
#             hyperparameters=new_hyperparameters,
#             presets='best_quality',
#             auto_stack=False,
#             # ds_args={
#             #     'memory_safe_fits': False,
#             # },
#             refit_full='best',
#             keep_only_best=True,  # Keep only the best model (and its ancestors)
#             num_cpus=num_cores,
#             save_space=True,
#             time_limit=60,  # only for testing
#         )
