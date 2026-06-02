import os
import pandas as pd

# %%
model_list = ['XGBoost']
all_feature_list = ['Sleep', 'Cov', 'Brain', 'CT', 'SA', 'Subcor', 'Sleep_Cov', 'Sleep_Cov_Brain', 'Sleep_Cov_CT',
                    'Sleep_Cov_SA', 'Sleep_Cov_Subcor', 'Sleep_Brain', 'Sleep_CT', 'Sleep_SA', 'Sleep_Subcor',
                    'Cov_Brain', 'Cov_CT', 'Cov_SA', 'Cov_Subcor', 'Sleep_Shuffle_Cov', 'Sleep_Cov_Subcor_Shuffle',
                    'Sleep_Shuffle_Cov_Subcor', 'Sleep_Shuffle_Subcor', 'Cov_Brain_Shuffle', 'Cov_Subcor_Shuffle']

# check scores.csv
print('\nStroop')
for model in model_list:
    for feature in all_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Stroop', 'SHIP_Trend', feature, 'scores.csv')
        if not os.path.exists(path):
            print(model, feature, 'missing scores.csv')

print('\nMemory')
for model in model_list:
    for feature in all_feature_list:
        path = os.path.join('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results', model,
                            'Memory', 'SHIP_Trend', feature, 'scores.csv')
        if not os.path.exists(path):
            print(model, feature, 'missing scores.csv')

# %%
# check whether each log.txt file has sentence: feature + 'completed successfully'.
print('\nStroop')
for model in model_list:
    for feature in all_feature_list:
        path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/' + model + '/combined_logs/SHIP/ml_pipeline/Stroop_' + feature + '_log.txt'
        with open(path, 'r') as f:
            content = f.read()
            if feature + ' completed successfully' not in content:
                print(f'Stroop, {model} {feature} not completed successfully')

print('\nMemory')
for model in model_list:
    for feature in all_feature_list:
        path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/' + model + '/combined_logs/SHIP/ml_pipeline/Memory_' + feature + '_log.txt'
        with open(path, 'r') as f:
            content = f.read()
            if feature + ' completed successfully' not in content:
                print(f'Memory, {model} {feature} not completed successfully')

# %%
# check SHAP
print('\nStroop')
for model in model_list:
    for feature in all_feature_list:
        path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/' + model + '/combined_logs/SHIP/SHAP/Stroop_' + feature + '_log.txt'
        if not os.path.exists(path):
            print(f'Stroop, {model} {feature} SHAP not completed successfully')
        else:
            with open(path, 'r') as f:
                content = f.read()
                if ' completed successfully' not in content:
                    print(f'Stroop, {model} {feature} SHAP not completed successfully')

print('\nMemory')
for model in model_list:
    for feature in all_feature_list:
        path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/' + model + '/combined_logs/SHIP/SHAP/Memory_' + feature + '_log.txt'
        if not os.path.exists(path):
            print(f'Memory, {model} {feature} SHAP not completed successfully')
        else:
            with open(path, 'r') as f:
                content = f.read()
                if ' completed successfully' not in content:
                    print(f'Memory, {model} {feature} SHAP not completed successfully')
