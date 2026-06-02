import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
# import advanced_ml_pipeline
# from advanced_ml_pipeline import AdMLPipeline
import AutoGluon_pipeline
from AutoGluon_pipeline import AdMLPipeline

import argparse
import pandas as pd
import numpy as np
import shap
import pickle
import joblib
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
    - Sleep_Cov_Brain_Shuffle: Uses sleep features, covariates, and brain imaging data (shuffled).
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Add htcondorcluster as an optional argument with a default value
parser.add_argument('htcondorcluster', type=str2bool, default=False,
                    help='''Whether to use HTCondorCluster for parallel processing. Default is False.''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target
num_cores = args.num_cores
htcondor_cluster = args.htcondorcluster

print(f"\nStarting AutoGluon pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
model_save_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/{target}/SHIP_Trend/'
# results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/{target}/SHIP_Trend/'

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
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(
        drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_SHIP_ml[APOE4] = df_SHIP_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

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
    df_Liege_ml[Thickness_Schaefer] = df_Liege_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(
        drop=True)
    df_Liege_ml[Area_DK] = df_Liege_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Area_Schaefer] = df_Liege_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_Liege_ml[APOE4] = df_Liege_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_Liege_ml[Thickness_DK] = df_Liege_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Thickness_Schaefer] = df_Liege_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Area_DK] = df_Liege_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Area_Schaefer] = df_Liege_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

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
df_Liege_COF_ml[Thickness_Schaefer] = df_Liege_COF_ml[Thickness_Schaefer].div(
    df_Liege_COF_ml[Thickness_Schaefer].sum(axis=1), axis=0)
df_Liege_COF_ml[Area_DK] = df_Liege_COF_ml[Area_DK].div(df_Liege_COF_ml[Area_DK].sum(axis=1), axis=0)
df_Liege_COF_ml[Area_Schaefer] = df_Liege_COF_ml[Area_Schaefer].div(df_Liege_COF_ml[Area_Schaefer].sum(axis=1), axis=0)
df_Liege_COF_ml[Subcortical] = df_Liege_COF_ml[Subcortical].div(df_Liege_COF_ml['EstimatedTotalIntraCranialVol'],
                                                                axis=0)

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
    df_Liege_COF_ml[Thickness_Schaefer] = df_Liege_COF_ml[Thickness_Schaefer].sample(frac=1,
                                                                                     random_state=33).reset_index(
        drop=True)
    df_Liege_COF_ml[Area_DK] = df_Liege_COF_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Area_Schaefer] = df_Liege_COF_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(
        drop=True)
    df_Liege_COF_ml[Subcortical] = df_Liege_COF_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_COF_ml[Subcortical] = df_Liege_COF_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_Liege_ml[APOE4] = df_Liege_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_Liege_COF_ml[Thickness_DK] = df_Liege_COF_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Thickness_Schaefer] = df_Liege_COF_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(
        drop=True)
    df_Liege_COF_ml[Area_DK] = df_Liege_COF_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Area_Schaefer] = df_Liege_COF_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(
        drop=True)
    df_Liege_COF_ml[Subcortical] = df_Liege_COF_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

"""
Data preprocessing for Liege COGNAP dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Liege_COGNAP_ml = utils.convert_units(df_Liege_COGNAP, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_Liege_COGNAP_ml[Thickness_DK] = df_Liege_COGNAP_ml[Thickness_DK].div(df_Liege_COGNAP_ml[Thickness_DK].sum(axis=1),
                                                                        axis=0)
df_Liege_COGNAP_ml[Thickness_Schaefer] = df_Liege_COGNAP_ml[Thickness_Schaefer].div(
    df_Liege_COGNAP_ml[Thickness_Schaefer].sum(axis=1), axis=0)
df_Liege_COGNAP_ml[Area_DK] = df_Liege_COGNAP_ml[Area_DK].div(df_Liege_COGNAP_ml[Area_DK].sum(axis=1), axis=0)
df_Liege_COGNAP_ml[Area_Schaefer] = df_Liege_COGNAP_ml[Area_Schaefer].div(df_Liege_COGNAP_ml[Area_Schaefer].sum(axis=1),
                                                                          axis=0)
df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].div(
    df_Liege_COGNAP_ml['EstimatedTotalIntraCranialVol'], axis=0)

df_Liege_COGNAP_ml['Memory_Test'] = df_Liege_COGNAP_ml['Memory_Test'] * 100
df_Liege_COGNAP_ml['1-back_acc'] = df_Liege_COGNAP_ml['1-back_acc'] * 100
df_Liege_COGNAP_ml['2-back_acc'] = df_Liege_COGNAP_ml['2-back_acc'] * 100
df_Liege_COGNAP_ml['3-back_acc'] = df_Liege_COGNAP_ml['3-back_acc'] * 100

# shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_Liege_COGNAP_ml[Sleep] = df_Liege_COGNAP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(
        drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_Liege_COGNAP_ml[Sleep] = df_Liege_COGNAP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_Liege_COGNAP_ml[Sleep] = df_Liege_COGNAP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_COGNAP_ml[Thickness_DK] = df_Liege_COGNAP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(
        drop=True)
    df_Liege_COGNAP_ml[Thickness_Schaefer] = df_Liege_COGNAP_ml[Thickness_Schaefer].sample(frac=1,
                                                                                           random_state=33).reset_index(
        drop=True)
    df_Liege_COGNAP_ml[Area_DK] = df_Liege_COGNAP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml[Area_Schaefer] = df_Liege_COGNAP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(
        drop=True)
    df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(
        drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(
        drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_Liege_COGNAP_ml[APOE4] = df_Liege_COGNAP_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_Liege_COGNAP_ml[Thickness_DK] = df_Liege_COGNAP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(
        drop=True)
    df_Liege_COGNAP_ml[Thickness_Schaefer] = df_Liege_COGNAP_ml[Thickness_Schaefer].sample(frac=1,
                                                                                           random_state=33).reset_index(
        drop=True)
    df_Liege_COGNAP_ml[Area_DK] = df_Liege_COGNAP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml[Area_Schaefer] = df_Liege_COGNAP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(
        drop=True)
    df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(
        drop=True)

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
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_KI_ml[Thickness_DK] = df_KI_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Thickness_Schaefer] = df_KI_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Area_DK] = df_KI_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Area_Schaefer] = df_KI_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Subcortical] = df_KI_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
# convert data types of columns. 'SEX' and 'APOE4' convert to object. Sleep, 'Age_at_Scan', 'BMI', TIV, and all imaging data to float64
columns_to_object = ['SEX', 'APOE4']
df_SHIP_ml[columns_to_object] = df_SHIP_ml[columns_to_object].astype('category')
df_Liege_ml[columns_to_object] = df_Liege_ml[columns_to_object].astype('category')
df_Liege_COF_ml[columns_to_object] = df_Liege_COF_ml[columns_to_object].astype('category')
df_Liege_COGNAP_ml[columns_to_object] = df_Liege_COGNAP_ml[columns_to_object].astype('category')
df_KI_ml[columns_to_object] = df_KI_ml[columns_to_object].astype('category')

# Convert specified columns to 'float64' data type
columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + targets
columns_to_float_KI = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + ['Memory_Test']
df_SHIP_ml[columns_to_float] = df_SHIP_ml[columns_to_float].astype('float64')
df_Liege_ml[columns_to_float] = df_Liege_ml[columns_to_float].astype('float64')
df_Liege_COF_ml[columns_to_float] = df_Liege_COF_ml[columns_to_float].astype('float64')
df_Liege_COGNAP_ml[columns_to_float] = df_Liege_COGNAP_ml[columns_to_float].astype('float64')
df_KI_ml[columns_to_float_KI] = df_KI_ml[columns_to_float_KI].astype('float64')

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
    'Sleep_Cov_Subcor_APOE_Shuffle': Sleep + Cov + Subcortical + APOE4,
    'Sleep_Cov_Brain_Shuffle': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical
}

y_dict = {
    'Stroop': 'Stroop_Test',
    'Memory': 'Memory_Test',
    'Stroop_rgo_age': 'Stroop_Test_rgo_age',
    'Memory_rgo_age': 'Memory_Test_rgo_age'
}

# %%
case_results_path = model_save_path + f"{feature_comb}/"
if not os.path.exists(case_results_path):
    os.makedirs(case_results_path)
predictor_save_path = model_save_path + f"{feature_comb}/" + 'autogluon/'

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml

# %%
# (self, X, y, df_train=None, num_cores=None, save_path=None, model=None)
ml_pipe_instance = AdMLPipeline(X, y, df_train, num_cores=num_cores, save_path=case_results_path)

print(f"\nStart CV resulting")
scores = ml_pipe_instance.run_autogluon_cross_val_iteration(stratified_label='Group_Age_SEX', htcondorcluster=htcondor_cluster)

# save scores
scores.to_csv(case_results_path + 'scores.csv', index=False)

print(f"\nCV resulting finished")

# %%
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

if 'Age_at_Scan' in X_dict[feature_comb] and 'rgo_age' in y:
    if 'Stroop_Test' in y:
        y_target = 'Stroop_Test'
    elif 'Memory_Test' in y:
        y_target = 'Memory_Test'

    # load transformer whole_age_poly.pkl and whole_age_lr.pkl
    # with open(case_results_path + 'whole_age_poly.pkl', 'rb') as file:
    #     whole_age_poly = pickle.load(file)
    # with open(case_results_path + 'whole_age_lr.pkl', 'rb') as file:
    #     whole_age_lr = pickle.load(file)
    whole_age_poly = joblib.load(case_results_path + 'whole_age_poly.pkl')
    whole_age_lr = joblib.load(case_results_path + 'whole_age_lr.pkl')

    y_true = df_train[y_target]
    age_train = df_train['Age_at_Scan'].values.reshape(-1, 1)
    age_train_poly = whole_age_poly.transform(age_train)
    y_true = y_true - whole_age_lr.predict(age_train_poly)
    print(f"Age adjusted {y_target} is used for evaluation.")
else:
    print(f"{y} is used for evaluation.")
    print(X_dict[feature_comb])
    y_true = df_train[y]

# %%
model = TabularPredictor.load(predictor_save_path)

y_oof_predict = model.predict_oof()
r2_oof = r2_score(y_true, y_oof_predict)
mse_oof = mean_squared_error(y_true, y_oof_predict)
mae_oof = mean_absolute_error(y_true, y_oof_predict)
r_corr_oof = np.corrcoef(y_true, y_oof_predict)[0, 1]
r_spearman_oof, _ = spearmanr(y_true, y_oof_predict)

# %%
# load results csv
results = pd.read_csv(case_results_path + 'results.csv', index_col=0)

results['Average CV Test MAE'] = -scores['test_neg_mean_absolute_error'].mean()
results['Average CV Test RMSE'] = -scores['test_neg_root_mean_squared_error'].mean()
results['Average CV Test R2'] = scores['test_r2'].mean()
results['Average CV Test Pearson r'] = scores['test_r_corr'].mean()
results['Average CV Test Spearman r'] = scores['test_spearmanr'].mean()

# after 'Average CV Test MAE' column, add 'OOF MAE'
loc_mae = results.columns.get_loc('Average CV Test MAE')
results.insert(loc_mae + 1, 'OOF MAE', mae_oof)
# after 'Average CV Test RMSE' column, add 'OOF RMSE'
loc_rmse = results.columns.get_loc('Average CV Test RMSE')
results.insert(loc_rmse + 1, 'OOF RMSE', np.sqrt(mse_oof))
# after 'Average CV Test R2' column, add 'OOF R2'
loc_r2 = results.columns.get_loc('Average CV Test R2')
results.insert(loc_r2 + 1, 'OOF R2', r2_oof)
# after 'Average CV Test Pearson r' column, add 'OOF Pearson r'
loc_r_corr = results.columns.get_loc('Average CV Test Pearson r')
results.insert(loc_r_corr + 1, 'OOF Pearson r', r_corr_oof)
# after 'Average CV Test Spearman r' column, add 'OOF Spearman r'
loc_r_spearman = results.columns.get_loc('Average CV Test Spearman r')
results.insert(loc_r_spearman + 1, 'OOF Spearman r', r_spearman_oof)

# save results
results.to_csv(case_results_path + 'cv_results.csv')

# %%
print(f"\nDone! AutoGluon pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print(f"Current date and time: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}\n")
