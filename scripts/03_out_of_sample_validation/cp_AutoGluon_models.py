import os
import sys

# %%
# results_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/'
# model_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/'
results_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon/'
model_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon_no_DK/'

# %%
target_list = ['Memory', 'Memory_rgo_age', 'Stroop', 'Stroop_rgo_age']
memory_target_list = ['Memory', 'Memory_rgo_age']
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

        original_model_folder = results_path + target + '/SHIP_Trend_no_DK/' + feature + '/autogluon/'
        save_model_folder = model_save_path + target + '/' + feature + '/'

        # Check if the original_model_folder exists, if not, report and skip this iteration
        if not os.path.exists(original_model_folder):
            print('The original_model_folder does not exist: ', original_model_folder)
            continue

        # Check if the save_model_folder exists, if not, create it
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder)

        # copy the original_model_folder to save_model_folder
        os.system('cp -r ' + original_model_folder + ' ' + save_model_folder)

        print(f'The model of {target}_{feature} has been saved to: ', save_model_folder)

# %%
# cp the target preprocessor for Memory_rgo_age and Stroop_rgo_age
for target in target_list:
    if '_rgo_age' in target:
        for feature in feature_list:
            original_files = results_path + target + '/SHIP_Trend_no_DK/' + feature + '/*.pkl'
            save_model_folder = model_save_path + target + '/' + feature + '/'

            # # Check if the original_files exist, if not, report and skip this iteration
            # if not os.path.exists(original_files):
            #     print('The original_files do not exist: ', original_files)
            #     continue
            #
            # # Check if the save_model_folder exists, if not, create it
            # if not os.path.exists(save_model_folder):
            #     os.makedirs(save_model_folder)

            # copy the original_files to save_model_folder
            os.system('cp ' + original_files + ' ' + save_model_folder)

            print(f'The preprocessor of {target}_{feature} has been saved to: ', save_model_folder)

# %%
# cp the shapiq results
# for feature in feature_list:
#     for target in memory_target_list:
#         # Check condition: if the target contains '_rgo_age', limit the features
#         if '_rgo_age' in target and feature not in ['Sleep_Cov', 'Sleep_Cov_Brain']:
#             continue  # Skip this iteration if the feature is not allowed
#
#         original_file = results_path + target + '/SHIP_Trend/' + feature + '/shapiq/ivs_SHIP_SHIP.pkl'
#         save_file = model_save_path + target + '/' + feature + '/shapiq/ivs_SHIP_SHIP.pkl'
#
#         # Check if the original_file exists, if not, report and skip this iteration
#         if not os.path.exists(original_file):
#             print('The original_file does not exist: ', original_file)
#             continue
#
#         # copy the original_file to save_file
#         os.system('cp ' + original_file + ' ' + save_file)
#         print(f'The shapiq results of {target}_{feature} have been saved to: ', save_file)
