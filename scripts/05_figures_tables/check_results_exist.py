import os

import numpy as np
import pandas as pd

# %%
save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/check_results_exist/'
results_saved_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/'
model_list = ['Dummy', 'Linear', 'Ridge', 'SVM-linear', 'SVM-rbf', 'rf', 'XGBoost', 'AutoGluon']
all_feature_combination_list = [
    'Sleep',
    'Cov',
    'Brain',
    'CT',
    'SA',
    'Subcor',
    'Sleep_Cov',
    'Sleep_Cov_Brain',
    'Sleep_Cov_CT',
    'Sleep_Cov_SA',
    'Sleep_Cov_Subcor',
    'Sleep_Brain',
    'Sleep_CT',
    'Sleep_SA',
    'Sleep_Subcor',
    'Cov_Brain',
    'Cov_CT',
    'Cov_SA',
    'Cov_Subcor',
    'Sleep_Shuffle_Cov',
    'Sleep_Cov_Subcor_Shuffle',
    'Sleep_Shuffle_Cov_Subcor',
    'Sleep_Shuffle_Subcor',
    'Cov_Brain_Shuffle',
    'Cov_Subcor_Shuffle',
    'Sleep_APOE',
    'Sleep_APOE_Shuffle',
    'Sleep_Cov_APOE',
    'Sleep_Cov_APOE_Shuffle',
    'Cov_APOE',
    'Cov_APOE_Shuffle',
    'Brain_APOE',
    'Brain_APOE_Shuffle',
    'Subcor_APOE',
    'Subcor_APOE_Shuffle',
    'Sleep_Cov_Brain_APOE',
    'Sleep_Cov_Brain_APOE_Shuffle',
    'Sleep_Cov_Subcor_APOE',
    'Sleep_Cov_Subcor_APOE_Shuffle'
]
main_feature_combination_list = [
    'Sleep',
    'Cov',
    'Sleep_Cov',
    'Sleep_Shuffle_Cov',
    'Brain',
    'Sleep_Cov_Brain',
    'Sleep_Cov_Brain_Shuffle'
]
APOE_feature_combination_list = [
    'Sleep_APOE',
    'Sleep_APOE_Shuffle',
    'Sleep_Cov_APOE',
    'Sleep_Cov_APOE_Shuffle',
    'Sleep_Cov_Brain_APOE',
    'Sleep_Cov_Brain_APOE_Shuffle'
]
Supplement_feature_combination_list = [
    'CT',
    'SA',
    'Subcor',
    'Sleep_Cov',
    'Sleep_Cov_Brain',
    'Sleep_Cov_CT',
    'Sleep_Cov_SA',
    'Sleep_Cov_Subcor',
    'Sleep_Brain',
    'Sleep_CT',
    'Sleep_SA',
    'Sleep_Subcor',
    'Cov_Brain',
    'Cov_CT',
    'Cov_SA',
    'Cov_Subcor',
    'Sleep_Shuffle_Cov',
    'Sleep_Cov_Subcor_Shuffle',
    'Sleep_Shuffle_Cov_Subcor',
    'Sleep_Shuffle_Subcor',
    'Cov_Brain_Shuffle',
    'Cov_Subcor_Shuffle',
]
target_list = [
    'Stroop',
    'Memory',
    'Stroop_rgo_age',
    'Memory_rgo_age'
]
training_set_list = [
    'SHIP_Trend',
    'SHIP_Trend_Liege'
]

# %%
# check for main feature combination results
# Open a text file to save the output
with open(save_path + 'main_feature_combination_results.txt', 'w') as f:
    sys.stdout = f

    print('\nChecking for main feature combination results...')
    dataframes = {}

    for target in target_list:
        print(f'\nPredict: {target}...')
        for training_set in training_set_list:
            print(f'\nTraining set is: {training_set}...')

            # Initialize an empty DataFrame
            df = pd.DataFrame(index=model_list, columns=main_feature_combination_list)

            for model in model_list:
                for feature_combination in main_feature_combination_list:
                    file_path = results_saved_path + f'{model}/{target}/{training_set}/{feature_combination}/scores.csv'
                    df.at[model, feature_combination] = os.path.exists(file_path)  # Set True if exists, False otherwise
                    if not os.path.exists(file_path):
                        print(f'{model}, {feature_combination} does not exist!')
                    else:
                        print(f'{model}, {feature_combination} exists!')

                    # Save DataFrame in a structured way in the dictionary
                    dataframes[(target, training_set)] = df
                    df.to_csv(f'{save_path}main_results_{target}_{training_set}.csv')

    sys.stdout = sys.__stdout__  # Reset standard output to the original state
print("The results have been saved to main_feature_combination_results.txt")

# %%
# check for APOE feature combination results
# Open a text file to save the output
with open(save_path + 'APOE_feature_combination_results.txt', 'w') as f:
    sys.stdout = f

    print('\nChecking for APOE feature combination results...')
    dataframes = {}

    for target in target_list:
        print(f'\nPredict: {target}...')
        for training_set in training_set_list:
            print(f'\nTraining set is: {training_set}...')

            # Initialize an empty DataFrame
            df = pd.DataFrame(index=model_list, columns=APOE_feature_combination_list)

            for model in model_list:
                for feature_combination in APOE_feature_combination_list:
                    file_path = results_saved_path + f'{model}/{target}/{training_set}/{feature_combination}/scores.csv'
                    df.at[model, feature_combination] = os.path.exists(file_path)  # Set True if exists, False otherwise
                    if not os.path.exists(file_path):
                        print(f'{model}, {feature_combination} does not exist!')
                    else:
                        print(f'{model}, {feature_combination} exists!')

                    # Save DataFrame in a structured way in the dictionary
                    dataframes[(target, training_set)] = df
                    df.to_csv(f'{save_path}APOE_results_{target}_{training_set}.csv')

    sys.stdout = sys.__stdout__  # Reset standard output to the original state
print("The results have been saved to APOE_feature_combination_results.txt")

# %%
# check for supplement feature combination results
# Open a text file to save the output
with open(save_path + 'supplement_feature_combination_results.txt', 'w') as f:
    sys.stdout = f

    print('\nChecking for supplement feature combination results...')
    dataframes = {}

    for target in target_list:
        print(f'\nPredict: {target}...')
        for training_set in training_set_list:
            print(f'\nTraining set is: {training_set}...')

            # Initialize an empty DataFrame
            df = pd.DataFrame(index=model_list, columns=Supplement_feature_combination_list)

            for model in model_list:
                for feature_combination in Supplement_feature_combination_list:
                    file_path = results_saved_path + f'{model}/{target}/{training_set}/{feature_combination}/scores.csv'
                    df.at[model, feature_combination] = os.path.exists(file_path)  # Set True if exists, False otherwise
                    if not os.path.exists(file_path):
                        print(f'{model}, {feature_combination} does not exist!')
                    else:
                        print(f'{model}, {feature_combination} exists!')

                    # Save DataFrame in a structured way in the dictionary
                    dataframes[(target, training_set)] = df
                    df.to_csv(f'{save_path}supplement_results_{target}_{training_set}.csv')

    sys.stdout = sys.__stdout__  # Reset standard output to the original state
print("The results have been saved to supplement_feature_combination_results.txt")
