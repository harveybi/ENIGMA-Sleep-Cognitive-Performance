import os
import sys

# %%
results_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/'
model_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/'

# %%
target_list = ['Memory', 'Memory_rgo_age', 'Stroop', 'Stroop_rgo_age']
feature_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
                'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
                'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle',
                'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle']

# %%
for feature in feature_list:
    for target in target_list:
        # Check condition: if the target contains '_rgo_age', limit the features
        if '_rgo_age' in target and feature not in ['Sleep_Cov', 'Sleep_Cov_Brain']:
            continue  # Skip this iteration if the feature is not allowed

        shap_path = model_save_path + target + '/' + feature + '/SHAP/'

        for filename in os.listdir(shap_path):
            # We look for files that start with "explainer_train_set_test_"
            if filename.startswith("explainer_train_set_test_") and filename.endswith(".pkl"):
                old_path = os.path.join(shap_path, filename)
                new_filename = filename.replace("explainer_", "explanation_", 1)  # Replace only the first occurrence
                new_path = os.path.join(shap_path, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} to {new_path}")

# %%
