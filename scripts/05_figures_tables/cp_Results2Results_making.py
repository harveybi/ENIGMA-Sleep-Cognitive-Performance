import os
import shutil

# %%
# Models / targets / features
model_list = ['Dummy', 'Linear', 'Ridge', 'SVM-linear', 'SVM-rbf', 'rf', 'XGBoost', 'AutoGluon']

target_list = ['Stroop', 'Memory', 'Stroop_rgo_age', 'Memory_rgo_age']
feature_list = [
    'Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
    'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
    'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle',
    'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle'
]

# Roots
original_folder = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results'
target_folder   = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/results_file'

# Constraints
skip_models_for_rgo = {'Dummy', 'Linear', 'Ridge', 'SVM-linear', 'SVM-rbf', 'rf'}
allowed_rgo_features = {'Sleep_Cov', 'Sleep_Cov_Brain'}

# %%
copied = 0

for model in model_list:
    for feature in feature_list:
        for target in target_list:

            # 1) These models do not have results for *_rgo_age
            if '_rgo_age' in target and model in skip_models_for_rgo:
                continue

            # 2) *_rgo_age only allowed for these features
            if '_rgo_age' in target and feature not in allowed_rgo_features:
                continue

            # 3) XGBoost uses SHIP_Trend_htcondor; others use SHIP_Trend
            trend_dir = 'SHIP_Trend_htcondor' if model == 'XGBoost' else 'SHIP_Trend'

            original_file = os.path.join(original_folder, model, target, trend_dir, feature, 'scores.csv')
            if not os.path.exists(original_file):
                print(f"[MISS] model={model} target={target} feature={feature}: not found -> {original_file}")
                continue

            target_path = os.path.join(target_folder, model, target, feature)
            os.makedirs(target_path, exist_ok=True)
            target_file = os.path.join(target_path, 'scores.csv')

            shutil.copy2(original_file, target_file)
            copied += 1
            print(f"[OK] Copied {original_file} -> {target_file}")

print(f"Done. Total files copied: {copied}")