import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import advanced_ml_pipeline
from advanced_ml_pipeline import AdMLPipeline
from autogluon.tabular import TabularPredictor

import argparse
import pandas as pd
import shap
import pickle
from datetime import datetime

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='SHAP for AutoGluon, SHIP_Trend dataset.')
parser.add_argument('feature_comb', type=str, help='''Name of the feature combination to be used.
    Available combinations are:
    - Sleep: Uses sleep features only.
    - Cov: Uses covariates only.
    - Sleep_Cov: Uses sleep features combined with covariates.
    - Sleep_Cov_Brain: Uses sleep features, covariates, and brain imaging data.
    - Sleep_Cov_CT: Uses sleep features, covariates, and cortical thickness from imaging data.
    - Sleep_Cov_SA: Uses sleep features, covariates, and surface area from imaging data.
    - Sleep_Cov_Subcor: Uses sleep features, covariates, and subcortical volumes.
    - Cov_Brain: Uses covariates combined with brain imaging data.
    - Cov_CT: Uses covariates and cortical thickness from imaging data.
    - Cov_SA: Uses covariates and surface area from imaging data.
    - Cov_Subcor: Uses covariates and subcortical volumes.
    - Sleep_Shuffle_Cov: Uses sleep features (shuffled) and covariates.
    - Sleep_Cov_Subcor_Shuffle: Uses sleep features, covariates, and subcortical volumes (shuffled).
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

print(f"\nStarting SHAP for AutoGluon pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/{target}/SHIP_Trend/'

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
2. Brain correction by brain size using internal data normalisation
3. NAI_Wordlist_Test transfer to accuracy
4. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']
df_SHIP_ml = utils.convert_units(df_SHIP, sleep_dur_cols, sleep_eff_cols)

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
if feature_comb == 'Cov_Brain_Shuffle':
    df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
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
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_ml[Thickness_DK] = df_Liege_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Thickness_Schaefer] = df_Liege_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Area_DK] = df_Liege_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_ml[Area_Schaefer] = df_Liege_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

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
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_COF_ml[Thickness_DK] = df_Liege_COF_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Thickness_Schaefer] = df_Liege_COF_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Area_DK] = df_Liege_COF_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COF_ml[Area_Schaefer] = df_Liege_COF_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_COF_ml[Subcortical] = df_Liege_COF_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

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
if feature_comb == 'Cov_Brain_Shuffle':
    df_Liege_COGNAP_ml[Thickness_DK] = df_Liege_COGNAP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml[Thickness_Schaefer] = df_Liege_COGNAP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml[Area_DK] = df_Liege_COGNAP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_Liege_COGNAP_ml[Area_Schaefer] = df_Liege_COGNAP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_Liege_COGNAP_ml[Subcortical] = df_Liege_COGNAP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

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
if feature_comb == 'Cov_Brain_Shuffle':
    df_KI_ml[Thickness_DK] = df_KI_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Thickness_Schaefer] = df_KI_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Area_DK] = df_KI_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_KI_ml[Area_Schaefer] = df_KI_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_KI_ml[Subcortical] = df_KI_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Define X and y
"""
X_dict = {
    'Sleep': Sleep,
    'Cov': Cov,
    'Sleep_Cov': Sleep + Cov,
    'Sleep_Cov_Brain': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_Cov_CT': Sleep + Cov + Thickness_DK + Thickness_Schaefer,
    'Sleep_Cov_SA': Sleep + Cov + Area_DK + Area_Schaefer,
    'Sleep_Cov_Subcor': Sleep + Cov + Subcortical,
    'Cov_Brain': Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Cov_CT': Cov + Thickness_DK + Thickness_Schaefer,
    'Cov_SA': Cov + Area_DK + Area_Schaefer,
    'Cov_Subcor': Cov + Subcortical,
    'Sleep_Shuffle_Cov': Sleep + Cov,
    'Sleep_Cov_Subcor_Shuffle': Sleep + Cov + Subcortical,
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


# create a wrapper class around AutoGluon to allow it to be called for prediction inside of the shap package
class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict(X)


case_results_path = results_path + f"{feature_comb}/"
predictor_save_path = case_results_path + 'autogluon/'

SHAP_path = case_results_path + 'SHAP/'
if not os.path.exists(SHAP_path):
    os.makedirs(SHAP_path)

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml

model = TabularPredictor.load(predictor_save_path)

# %%
ag_wrapper = AutogluonWrapper(model, X)
explainer_train_set = shap.explainers.Permutation(ag_wrapper.predict, df_train[X], max_evals=2000)

# %%
# explanation for the training set
explanation_train_set_train = explainer_train_set(df_SHIP_ml[X])
with open(SHAP_path + 'explainer_train_set_train.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_train, f)
# explanation for the test set
explanation_train_set_test_Liege = explainer_train_set(df_Liege_ml[X])
with open(SHAP_path + 'explainer_train_set_test_Liege.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_test_Liege, f)

explanation_train_set_test_Liege_COF = explainer_train_set(df_Liege_COF_ml[X])
with open(SHAP_path + 'explainer_train_set_test_Liege_COF.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_test_Liege_COF, f)

# %%
explanation_train_set_test_Liege_COGNAP = explainer_train_set(df_Liege_COGNAP_ml[X])
with open(SHAP_path + 'explainer_train_set_test_Liege_COGNAP.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_test_Liege_COGNAP, f)

# %%
if target == 'Memory':
    explanation_train_set_test_KI = explainer_train_set(df_KI_ml[X])
    with open(SHAP_path + 'explainer_train_set_test_KI.pkl', 'wb') as f:
        pickle.dump(explanation_train_set_test_KI, f)

# %%
ml_pipe_instance = AdMLPipeline(X, y)

ml_pipe_instance.shap_explain(explanation_train_set_train, SHAP_path, 'train_set_train')
ml_pipe_instance.shap_explain(explanation_train_set_test_Liege, SHAP_path, 'train_set_test_Liege')
ml_pipe_instance.shap_explain(explanation_train_set_test_Liege_COF, SHAP_path, 'train_set_test_Liege_COF')
ml_pipe_instance.shap_explain(explanation_train_set_test_Liege_COGNAP, SHAP_path, 'train_set_test_Liege_COGNAP')
if target == 'Memory':
    ml_pipe_instance.shap_explain(explanation_train_set_test_KI, SHAP_path, 'train_set_test_KI')

print("SHAP explanation completed.\n")

# %%
print(f"\nDone! SHAP AutoGluon pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
