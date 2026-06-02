import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import AutoGluon_pipeline
from AutoGluon_pipeline import AdMLPipeline
from autogluon.tabular import TabularPredictor

import argparse
import pandas as pd
import shap
import pickle
from joblib import load
from datetime import datetime

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='SHAP for AutoGluon, VETSA dataset.')
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

print(f"\nStarting SHAP for AutoGluon pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'  # where the data is saved
trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon_no_DK/{target}/{feature_comb}/'  # where the trained models are saved

df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test', 'Memory_Test'])

if 'Memory' in target:
    if 'rgo_age' in target:
        trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon_no_DK/Memory_rgo_age/{feature_comb}/'
    else:
        trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon_no_DK/Memory/{feature_comb}/'  # where the trained models are saved
else:
    trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon_no_DK/{target}/{feature_comb}/'  # where the trained models are saved

df_VETSA = pd.read_csv(data_save_path + 'VETSA_dataset_renamed_target_cleaned.csv')
df_VETSA = df_VETSA.dropna(subset=['Stroop_Test', 'Memory_Digit_Test', 'Memory_Letter_Test'])

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

# %%
"""
Data preprocessing for SHIP_Trend dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. NAI_Wordlist_Test transfer to accuracy
4. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
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
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

columns_to_object = ['SEX', 'APOE4']
df_SHIP_ml[columns_to_object] = df_SHIP_ml[columns_to_object].astype('category')

columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + targets
df_SHIP_ml[columns_to_float] = df_SHIP_ml[columns_to_float].astype('float64')

# %%
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
    'Brain': Thickness_Schaefer + Area_Schaefer + Subcortical,
    'Sleep_Cov_Brain': Sleep + Cov + Thickness_Schaefer + Area_Schaefer + Subcortical,
    'Sleep_Cov_Brain_Shuffle': Sleep + Cov + Thickness_Schaefer + Area_Schaefer + Subcortical,
    'Sleep_APOE': Sleep + APOE4,
    'Sleep_APOE_Shuffle': Sleep + APOE4,
    'Sleep_Cov_APOE': Sleep + Cov + APOE4,
    'Sleep_Cov_APOE_Shuffle': Sleep + Cov + APOE4,
    'Sleep_Cov_Brain_APOE': Sleep + Cov + Thickness_Schaefer + Area_Schaefer + Subcortical + APOE4,
    'Sleep_Cov_Brain_APOE_Shuffle': Sleep + Cov + Thickness_Schaefer + Area_Schaefer + Subcortical + APOE4
}

y_dict = {
    'Stroop': 'Stroop_Test',
    'Memory': 'Memory_Test',
    'Stroop_rgo_age': 'Stroop_rgo_age',
    'Memory_rgo_age': 'Memory_rgo_age'
}

# %%
'''
SHAP explanation
'''
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

# load model
print(f"\nLoad model")
case_results_path = trained_models_path
predictor_save_path = case_results_path + 'autogluon/'
model = TabularPredictor.load(predictor_save_path)


X = X_dict[feature_comb]
if 'Memory' in target:
    if 'rgo_age' in target:
        target = 'Memory_rgo_age'
    else:
        target = 'Memory'

y = y_dict[target]
df_train = df_SHIP_ml
df_test = df_VETSA_ml

# %%
if 'Age_at_Scan' in X and 'rgo_age' in y:
    whole_age_poly_saved_path = case_results_path + 'whole_age_poly.pkl'
    whole_age_poly = load(whole_age_poly_saved_path)

    whole_age_lr_saved_path = case_results_path + 'whole_age_lr.pkl'
    whole_age_lr = load(whole_age_lr_saved_path)
else:
    whole_age_poly = None
    whole_age_lr = None

rgo_age = False
if 'Age_at_Scan' in X and 'rgo_age' in y:
    y = y.replace('_rgo_age', '_Test')
    rgo_age = True
    X.remove('Age_at_Scan')

y_train = df_train[y].values

if rgo_age:
    age_train = df_train['Age_at_Scan'].values.reshape(-1, 1)

    age_train_poly = whole_age_poly.transform(age_train)

    y_train = y_train - whole_age_lr.predict(age_train_poly)

    y_true_train = y_train
    df_train[y] = y_train

else:
    y_true_train = y_train

# %%
ag_wrapper = AutogluonWrapper(model, X)

clustering = shap.utils.hclust(df_train[X], y_true_train)
masker = shap.maskers.Partition(df_train[X], clustering=clustering)

print("\nStart SHAP explainer")
explainer_train_set = shap.Explainer(ag_wrapper.predict, masker=masker, algorithm='auto', feature_names=X, seed=33)  # algorithm='permutation', max_evals=2000

# %%
SHAP_path = case_results_path + 'SHAP/'
if not os.path.exists(SHAP_path):
    os.makedirs(SHAP_path)

# save the explainer
with open(SHAP_path + 'explainer_train_set_VETSA.pkl', 'wb') as f:
    pickle.dump(explainer_train_set, f)

print("SHAP explainer saved.\n")

# %%
print("\nStart SHAP explanation")
# explanation for the test set
explanation_train_set_test_VETSA = explainer_train_set(df_test[X], max_evals=2000)
with open(SHAP_path + 'explanation_train_set_test_VETSA.pkl', 'wb') as f:
    pickle.dump(explanation_train_set_test_VETSA, f)

print("SHAP explanation completed.\n")

# %%
print(f"\nDone! SHAP AutoGluon pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
