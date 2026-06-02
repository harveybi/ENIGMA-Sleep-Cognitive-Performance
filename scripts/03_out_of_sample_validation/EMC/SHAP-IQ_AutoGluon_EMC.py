import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/lib/')
import utils
import AutoGluon_pipeline
from AutoGluon_pipeline import AdMLPipeline
from autogluon.tabular import TabularPredictor

import argparse
import pandas as pd
import shapiq
import pickle
from joblib import dump, load, Parallel, delayed, parallel_config
from joblib_htcondor import register_htcondor
from datetime import datetime

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='SHAP-IQ for AutoGluon, EMC dataset.')
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
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target

print(f"\nStarting SHAP-IQ for AutoGluon pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'  # where the data is saved
trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/{target}/{feature_comb}/'  # where the trained models are saved

df_EMC = pd.read_csv(data_save_path + 'EMC_dataset_renamed_target_cleaned.csv')
df_EMC = df_EMC.dropna(subset=['Stroop_Test'])

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
Data preprocessing for EMC dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_EMC_ml = utils.convert_units(df_EMC, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_EMC_ml[Thickness_DK] = df_EMC_ml[Thickness_DK].div(df_EMC_ml[Thickness_DK].sum(axis=1), axis=0)
df_EMC_ml[Thickness_Schaefer] = df_EMC_ml[Thickness_Schaefer].div(df_EMC_ml[Thickness_Schaefer].sum(axis=1),
                                                                      axis=0)
df_EMC_ml[Area_DK] = df_EMC_ml[Area_DK].div(df_EMC_ml[Area_DK].sum(axis=1), axis=0)
df_EMC_ml[Area_Schaefer] = df_EMC_ml[Area_Schaefer].div(df_EMC_ml[Area_Schaefer].sum(axis=1), axis=0)
df_EMC_ml[Subcortical] = df_EMC_ml[Subcortical].div(df_EMC_ml['EstimatedTotalIntraCranialVol'], axis=0)

# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_EMC_ml[Sleep] = df_EMC_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_EMC_ml[Subcortical] = df_EMC_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_EMC_ml[Sleep] = df_EMC_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_EMC_ml[Sleep] = df_EMC_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_EMC_ml[Thickness_DK] = df_EMC_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_EMC_ml[Thickness_Schaefer] = df_EMC_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_EMC_ml[Area_DK] = df_EMC_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_EMC_ml[Area_Schaefer] = df_EMC_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_EMC_ml[Subcortical] = df_EMC_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_EMC_ml[Subcortical] = df_EMC_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_EMC_ml[APOE4] = df_EMC_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_EMC_ml[Thickness_DK] = df_EMC_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_EMC_ml[Thickness_Schaefer] = df_EMC_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_EMC_ml[Area_DK] = df_EMC_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_EMC_ml[Area_Schaefer] = df_EMC_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_EMC_ml[Subcortical] = df_EMC_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# convert data types of columns. 'SEX' and 'APOE4' convert to object. Sleep, 'Age_at_Scan', 'BMI', TIV, and all imaging data to float64
columns_to_object = ['SEX', 'APOE4']
df_EMC_ml[columns_to_object] = df_EMC_ml[columns_to_object].astype('category')

# Convert specified columns to 'float64' data type
columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical
df_EMC_ml[columns_to_float] = df_EMC_ml[columns_to_float].astype('float64')

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
}

# %%
'''
SHAP-IQ explanation
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
y = y_dict[target]
df_test = df_EMC_ml

y_train = df_train[y].values

# %%
shapiq_path = case_results_path + 'shapiq/'
if not os.path.exists(shapiq_path):
    os.makedirs(shapiq_path)

ag_wrapper = AutogluonWrapper(model, X)

test_sample_size = df_test.shape[0]

explainer_file = os.path.join(shapiq_path, "explainer_SHIP_EMC.joblib")

# -------- load --------
payload = joblib.load(explainer_file)
explainer_train_set = cloudpickle.loads(payload)
print("Loaded explainer.")

# %%
print("\nStart of explanation")
register_htcondor("INFO")

def explain_instance(x):
    # This function calls the explainer's explain method on a single instance.
    return explainer_train_set.explain(x, budget=1024)


with parallel_config(backend="loky", n_jobs=-1, verbose=10):
    ivs = Parallel()(
        delayed(explain_instance)(df_test[X].values[i, :]) for i in range(test_sample_size)
    )
print(f"\nExplanation completed\n")

with open(shapiq_path + 'ivs_SHIP_EMC.pkl', 'wb') as f:
    pickle.dump(ivs, f)

print("\nNote, the SHAP-IQ value based on SHIP-Trend to EMC is saved, but no plots were generated.\n")

# %%
print(f"\nDone! SHAP-IQ AutoGluon pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
