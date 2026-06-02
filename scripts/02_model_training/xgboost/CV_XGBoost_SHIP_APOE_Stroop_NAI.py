import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')

import utils
import CV_XGBoost_pipeline
from CV_XGBoost_pipeline import AdMLPipeline

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
parser = argparse.ArgumentParser(description='XGBoost, SHIP_Trend dataset.')
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

parser.add_argument("repeat", type=int, help="Repeat index for nested CV")

parser.add_argument("fold", type=int, help="Fold index for nested CV")

# Add num_cores as an optional argument with a default value
parser.add_argument('num_cores', type=int, default=1,
                    help='Number of cores to be used for parallel processing. Default is 1.')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target
repeat = args.repeat
fold = args.fold
num_cores = args.num_cores

print(f"\nStarting XGBoost pipeline for {target} prediction with feature combination {feature_comb}.\n")
start_time = datetime.now()

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/CV_XGBoost/{target}/SHIP_Trend/'

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
print(f"\nStart model training, repeat: {repeat}, fold: {fold}")
train_start_time = datetime.now()

case_results_path = results_path + f"{feature_comb}/R{repeat + 1}F{fold + 1}/"
if not os.path.exists(case_results_path):
    os.makedirs(case_results_path)

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml

# X, y, df_train=None, num_cores=None, save_path=None
ml_pipe_instance = AdMLPipeline(X, y, df_train, num_cores, case_results_path)

score = ml_pipe_instance.xgboost_pipe(repeat, fold, stratified_label='Group_Age_SEX')

print(score)
# save score
score.to_csv(case_results_path + 'score.csv', index=False)

# calculate used time
train_end_time = datetime.now()

# time format: hours, minutes, seconds
print(f"\nModel training completed. Time used: {train_end_time - train_start_time}")

print("\nModel performance:")
print(score)

# %%
print(f"\nDone! XGBoost pipeline for {target} prediction by {feature_comb} repeat: {repeat} fold: {fold} completed successfully!")
# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
# print total execution time by hours and minutes
execution_time = datetime.now() - start_time
print("Execution Time: ", execution_time)
