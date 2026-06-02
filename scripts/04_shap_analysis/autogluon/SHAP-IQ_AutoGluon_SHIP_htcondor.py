import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
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

import matplotlib.pyplot as plt

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='SHAP-IQ for AutoGluon, SHIP_Trend dataset.')
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

print(f"\nStarting SHAP-IQ for AutoGluon pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/{target}/SHIP_Trend/'

# %%
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
shapiq explanation
'''
# create a wrapper class around AutoGluon to allow it to be called for prediction inside of the shapiq package
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

# load data and model
X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml

model = TabularPredictor.load(predictor_save_path)

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

    y_true = y_train
    df_train[y] = y_train

    print(f"\ny_true range: {y_true.min()} - {y_true.max()}\n")

else:
    y_true = y_train

# %%
shapiq_path = case_results_path + 'shapiq/'
if not os.path.exists(shapiq_path):
    os.makedirs(shapiq_path)

ag_wrapper = AutogluonWrapper(model, X)

# get sample size from the training data
sample_size = df_train.shape[0]
imputer = shapiq.MarginalImputer(
    model=ag_wrapper.predict, data=df_train[X].values, sample_size=sample_size, random_state=33
)

if 'Brain' in feature_comb:
    approx = "svarm"
else:
    approx = "auto"

print(f"\nStart of explainer\n")
explainer_train_set = shapiq.TabularExplainer(
    model=ag_wrapper.predict,
    data=df_train[X].values,
    imputer=imputer,
    approximator=approx,
    random_state=33
)
print(f"\nExplainer created\n")
# explainer_train_set = shapiq.TabularExplainer(ag_wrapper.predict, data=df_train[X].values, random_state=33)

# %%
print(f"\nStart of explanation\n")
register_htcondor("INFO")

def explain_instance(x):
    # This function calls the explainer's explain method on a single instance.
    return explainer_train_set.explain(x, budget=1024)

# if 'Brain' in feature_comb, request 2GB disk
if 'Brain' in feature_comb:
    memory_request = "4GB"
else:
    memory_request = "2GB"

with parallel_config(
        backend="htcondor",
        pool="head2.htc.inm7.de",
        n_jobs=-1,
        request_cpus=1,
        request_disk="1GB",
        request_memory=memory_request,
        shared_data_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/joblib_htcondor/SHAP-IQ",
        log_dir_prefix="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/logs/SHAP-IQ",
        throttle=[100],
        export_metadata=True
):
    # ivs = explainer_train_set.explain_X(df_train[X].values, budget=1024, n_jobs=-1)
    # ivs = explainer_train_set.explain_X(df_train[X].values[:10, :], budget=1024)
    ivs = Parallel()(
        delayed(explain_instance)(df_train[X].values[i, :]) for i in range(sample_size)
    )
print(f"\nExplanation completed\n")

# save ivs by pickle
with open(shapiq_path + 'ivs_SHIP_SHIP.pkl', 'wb') as f:
    pickle.dump(ivs, f)

# %%
# load ivs by pickle
# with open(shapiq_path + 'ivs.pkl', 'rb') as f:
#     ivs = pickle.load(f)

# %%
# shapiq.plot.bar_plot(ivs, feature_names=X)
# plt.show()

# %%
# # first subject
# interaction_values = ivs[2]

# %%
# # network plot
# _ = shapiq.network_plot(
#     first_order_values=interaction_values.get_n_order_values(1),
#     second_order_values=interaction_values.get_n_order_values(2),
#     feature_names=X,
# )
# plt.show()
# plt.savefig(shapiq_path + "network_plot.png", dpi=300)
#
# # %%
# # force plot
# _ = shapiq.stacked_bar_plot(
#     interaction_values=interaction_values,
#     feature_names=X,
# )
# plt.savefig(shapiq_path + "stacked_bar_plot.png", dpi=300)
#
# # %%
# # force plot
# interaction_values.plot_force(feature_names=X)
# plt.savefig(shapiq_path + "force_plot.png", dpi=300)
#
# # %%
# # waterfall plot
# # interaction_values.plot_waterfall(feature_names=X)
# # plt.savefig(shapiq_path + "waterfall_plot.png", dpi=300)
#
# # %%
# # upset plot
# _ = shapiq.upset_plot(
#     interaction_values=interaction_values,
#     feature_names=X,
# )
# plt.savefig(shapiq_path + "upset_plot.png", dpi=300)

# %%
# save the explainer
# with open(shapiq_path + 'explainer_train_set.pkl', 'wb') as f:
#     pickle.dump(explainer_train_set, f)
#
# print("shapiq explainer saved.\n")
#
# # %%
# print("\nStart shapiq explanation")
# # explanation for the training set
# explanation_train_set_train = explainer_train_set(df_train[X], max_evals=2000)
# with open(shapiq_path + 'explanation_train_set_train.pkl', 'wb') as f:
#     pickle.dump(explanation_train_set_train, f)
#
# print("shapiq explanation completed.\n")

# %%
# ml_pipe_instance = AdMLPipeline(X, y)
#
# ml_pipe_instance.shapiq_explain(explanation_train_set_train, shapiq_path, 'train_set_train')

print("\nNote, the SHAP-IQ value based on SHIP-Trend is saved, but no plots were generated.\n")

# %%
print(f"\nDone! SHAP-IQ AutoGluon pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
