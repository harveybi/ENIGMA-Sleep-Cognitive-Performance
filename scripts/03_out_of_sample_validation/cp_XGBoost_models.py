import os
import sys

# %%
results_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/'
model_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/XGBoost/'

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

        original_model_folder = results_path + target + '/SHIP_Trend_htcondor/' + feature
        save_model_folder = model_save_path + target + '/' + feature + '/'

        # Check if the original_model_folder exists, if not, report and skip this iteration
        if not os.path.exists(original_model_folder):
            print('The original_model_folder does not exist: ', original_model_folder)
            continue

        # Create the save_model_folder if it does not exist
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder)

        original_model_file = original_model_folder + '/xgboost_model.json'
        save_model_file = save_model_folder + 'xgboost_model.json'

        original_preprocessor_file = original_model_folder + '/whole_X_train_preprocessor.pkl'
        save_preprocessor_file = save_model_folder + 'whole_X_train_preprocessor.pkl'

        original_cat_file = original_model_folder + '/cat_dtypes.pkl'
        save_cat_file = save_model_folder + 'cat_dtypes.pkl'

        # copy files
        os.system('cp ' + original_model_file + ' ' + save_model_file)
        os.system('cp ' + original_preprocessor_file + ' ' + save_preprocessor_file)
        os.system('cp ' + original_cat_file + ' ' + save_cat_file)

        if '_rgo_age' in target:
            original_age_lr_file = original_model_folder + '/whole_age_lr.pkl'
            original_age_poly_file = original_model_folder + '/whole_age_poly.pkl'
            save_age_lr_file = save_model_folder + 'whole_age_lr.pkl'
            save_age_poly_file = save_model_folder + 'whole_age_poly.pkl'

            # copy age files
            os.system('cp ' + original_age_lr_file + ' ' + save_age_lr_file)
            os.system('cp ' + original_age_poly_file + ' ' + save_age_poly_file)

        print(f'Copied model and preprocessor for target: {target}, feature: {feature}')
