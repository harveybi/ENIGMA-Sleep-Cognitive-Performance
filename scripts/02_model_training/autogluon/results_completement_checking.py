import os
import sys

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/'
log_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/combined_logs/SHIP/'

# %%
feature_comb_list = [
    'Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
    'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
    'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle',
    'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle'
]
target_list = [
    'Stroop', 'Memory', 'Stroop_rgo_age', 'Memory_rgo_age'
]

# %%
"""
Check SHAP results for AutoGluon
"""
# Lists to track issues
not_exist_logs = []
not_completed_logs = []

for feature in feature_comb_list:
    for target in target_list:
        # Check condition: if the target contains '_rgo_age', limit the features
        if '_rgo_age' in target and feature not in ['Sleep_Cov', 'Sleep_Cov_Brain']:
            continue  # Skip this iteration if the feature is not allowed

        log_file_name = f'{target}_{feature}_log.txt'
        log_file_path = os.path.join(log_path, "SHAP", log_file_name)

        # Check if the log file exists
        if not os.path.exists(log_file_path):
            not_exist_logs.append(log_file_name)
            continue

        # Check if the log file contains 'completed successfully'
        with open(log_file_path, 'r') as log_file:
            log_content = log_file.read()
            if 'completed successfully' not in log_content:
                not_completed_logs.append(log_file_name)

# Print results
if not_exist_logs:
    print("The following log files do not exist:")
    print("\n".join(not_exist_logs))
else:
    print("All log files exist.")

if not_completed_logs:
    print("\nThe following log files did not complete successfully:")
    print("\n".join(not_completed_logs))
else:
    print("\nAll scripts completed successfully.")

