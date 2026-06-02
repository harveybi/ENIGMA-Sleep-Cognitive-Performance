import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import XGBoost_pipeline
from XGBoost_pipeline import AdMLPipeline
import xgboost as xgb

import argparse
import pandas as pd
import pickle
import shap
from joblib import load
from datetime import datetime


import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='SHAP for XGBoost, SHIP_Trend dataset.')
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
    - Memory: Memory_Test
    - Stroop_rgo_age: Stroop_rgo_age
    - Memory_rgo_age: Memory_rgo_age
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target

print(f"\nStarting SHAP for XGBoost pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/{target}/SHIP_Trend_htcondor/'

df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test', 'Memory_Test'])

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

columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + targets
df_SHIP_ml[columns_to_float] = df_SHIP_ml[columns_to_float].astype('float64')

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
    'Memory': 'Memory_Test',
    'Stroop_rgo_age': 'Stroop_rgo_age',
    'Memory_rgo_age': 'Memory_rgo_age'
}

# %%
'''
SHAP explanation
'''
print("\nStart SHAP explanation")

# load data and model
case_results_path = results_path + f"{feature_comb}/"

columns_to_int = ['SEX', 'APOE4']
df_SHIP_ml[columns_to_int] = df_SHIP_ml[columns_to_int].fillna(0).astype('int64')
# load the saved categorical coder
with open(case_results_path + "cat_dtypes.pkl", "rb") as f:
    cat_dtypes = pickle.load(f)
for col, dtype in cat_dtypes.items():
    df_SHIP_ml[col] = pd.Categorical(df_SHIP_ml[col], dtype.categories)

# %%
SHAP_path = case_results_path + 'SHAP/'
if not os.path.exists(SHAP_path):
    os.makedirs(SHAP_path)

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml

model = xgb.XGBRegressor()
model.load_model(case_results_path + 'xgboost_model.json')

# %%
# data apply the same preprocessing as the training set
# Load the transformer using joblib
preprocessors_saved_path = case_results_path + 'whole_X_train_preprocessor.pkl'
whole_X_train_preprocessor = load(preprocessors_saved_path)

# If 'rgo_age' in y, also load whole_age_poly and whole_age_lr
if 'Age_at_Scan' in X and 'rgo_age' in y:
    whole_age_poly_saved_path = case_results_path + 'whole_age_poly.pkl'
    whole_age_poly = load(whole_age_poly_saved_path)

    whole_age_lr_saved_path = case_results_path + 'whole_age_lr.pkl'
    whole_age_lr = load(whole_age_lr_saved_path)
else:
    whole_age_poly = None
    whole_age_lr = None

X_train = whole_X_train_preprocessor.transform(df_train[X])
X_train.columns = [column.split('__')[-1] for column in X_train.columns]

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

    y_true = y_train
    print(f"\ny_true range: {y_true.min()} - {y_true.max()}\n")

else:
    y_true = y_train

# make sure the X_test columns order is same as the feature order in the model
# make the X_test columns order same as the self.X order
X_train = X_train[model.feature_names_in_]

# %%
print(f"SHAP version: {shap.__version__}")
print(f"XGBoost version: {xgb.__version__}")
print(f"Pandas version: {pd.__version__}")

print("\nGet explainer")
# explainer_train_set = shap.TreeExplainer(model, data=X_train, feature_perturbation='tree_path_dependent')
explainer_train_set = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent', feature_names=model.feature_names_in_)

print("\nGet explanation")
print("--- Data Check Before Explanation Calculation ---")
print(f"X_train type: {type(X_train)}")
if isinstance(X_train, pd.DataFrame):
    print("X_train DataFrame Info:")
    X_train.info() # Check all column types
else:
    print(f"X_train is not a DataFrame (type: {type(X_train)}), this could be an issue.")
print("--- End Data Check ---")

with open(SHAP_path + 'explainer_train_set.pkl', 'wb') as f:
    pickle.dump(explainer_train_set, f)

# --- Manual DMatrix Workaround (Corrected) ---
print("\nGet explanation (Workaround: Manual DMatrix)")

# Ensure X_train is the DataFrame with correct columns and dtypes
print("--- Creating DMatrix Manually (Corrected) ---")
print("X_train dtypes for DMatrix creation:")
X_train.info() # X_train is the aligned DataFrame

try:
    # --- FIX: Convert feature names to list ---
    if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        # Convert numpy array (or other sequence) to a list of strings
        feature_names_list = list(map(str, model.feature_names_in_))
        print(f"Using feature names from model: {feature_names_list}")
    else:
        # Fallback if feature names aren't available on the model
        print("Warning: Model has no 'feature_names_in_'. Using DataFrame columns as feature names.")
        feature_names_list = X_train.columns.tolist()

    # Ensure X_train columns match the feature names list order before creating DMatrix
    print("Reordering X_train columns to match feature_names_list for DMatrix...")
    X_train_reordered = X_train[feature_names_list]

    # Create DMatrix explicitly enabling categorical features and using list of names
    dtrain_for_shap = xgb.DMatrix(X_train_reordered, # Use reordered data
                                  enable_categorical=True,
                                  feature_names=feature_names_list) # Pass the list
    print("Manual DMatrix created successfully.")

    # --- Now try using the manual DMatrix with SHAP ---
    # THIS IS THE CRITICAL TEST
    try:
        print("\nAttempting explanation calculation using the manual DMatrix...")
        explanation_train_set_train = explainer_train_set(dtrain_for_shap) # Pass DMatrix
        print(">>> Explanation calculation with manual DMatrix SUCCEEDED! <<<")

        # If successful, save the result
        explanation_filename = SHAP_path + 'explanation_tdep_manual_dmatrix.pkl'
        with open(explanation_filename, 'wb') as f:
            pickle.dump(explanation_train_set_train, f)
        print(f"Explanation object saved to {explanation_filename}")
        print(f"SHAP values shape: {explanation_train_set_train.values.shape}")

    # --- Handle potential errors when passing DMatrix to SHAP ---
    except TypeError as te:
        # This error means SHAP doesn't accept DMatrix input here
        print(f"\nTypeError: SHAP explainer call failed - it likely does not accept DMatrix input directly.")
        print(f"Error details: {te}")
        print("--> Manual DMatrix workaround failed because SHAP expects DataFrame/NumPy array.")
        print("--> This points back to a potential issue in SHAP/XGBoost interaction for this version combination.")
        # Consider raising error or proceeding to KernelExplainer fallback here
        raise te
    except ValueError as ve:
         # Check if it's the *original* ValueError about enable_categorical!
         print(f"\nValueError during explanation with manual DMatrix: {ve}")
         if "enable_categorical" in str(ve) and "must be set to `True`" in str(ve):
             print("!!! The ORIGINAL error about 'enable_categorical' persists EVEN when passing a manually created DMatrix with the flag set!")
             print("!!! This strongly suggests a bug or deep incompatibility in SHAP 0.47.2 / XGBoost 2.0.3.")
         else:
             print("--> A different ValueError occurred.")
         # Re-raise the error to investigate further or stop
         raise ve
    except Exception as e:
        print(f"\nUnexpected error during explanation with manual DMatrix: {e}")
        raise e # Re-raise to stop script

# --- Catch errors during DMatrix creation itself ---
except Exception as e:
    print(f"\nERROR creating manual DMatrix (even after fix attempt): {e}")
    # This part catches the DMatrix creation error if it still happens for some reason
    raise e

# %%
# load the explanation
with open(SHAP_path + 'explanation_tdep_manual_dmatrix.pkl', 'rb') as f:
    explanation_train_set_train = pickle.load(f)

# SHAP bar plot with clustered features
clustering = shap.utils.hclust(X_train[X], y_true)
shap.plots.bar(explanation_train_set_train, clustering=clustering, show=False)
# save the plot
plt.savefig(SHAP_path + f"SHAP_bar_{target}_{feature_comb}.png")
plt.close()

# %%
print(f"\nDone! SHAP AutoGluon pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
