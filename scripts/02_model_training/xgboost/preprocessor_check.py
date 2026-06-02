import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')  # change to the path where the lib folder is located
import utils
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import pandas as pd
import pickle
from datetime import datetime

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
result_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/'

target_list = ['Stroop', 'Stroop_rgo_age', 'Memory', 'Memory_rgo_age']

feature_comb_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov', 'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
                     'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle', 'Sleep_Cov_Brain_APOE',
                     'Sleep_Cov_Brain_APOE_Shuffle']

# %%
result_path_1 = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/Stroop/SHIP_Trend_test/'

# get the list of all files in the directory
feature_list = os.listdir(result_path_1)

# %%
for feature in feature_list:
    transformer_save_path = result_path_1 + feature + '/whole_X_train_preprocessor.pkl'
    with open(transformer_save_path, 'rb') as f:
        whole_X_train_preprocessor = pickle.load(f)

    print(f'{feature}\'s preprocessor type is {type([whole_X_train_preprocessor])}')

# %%
import xgboost as xgb

for feature in feature_list:
    model = xgb.XGBRegressor()
    model.load_model(result_path_1 + feature + '/xgboost_model.json')
    # print feature inside the model
    print(f'{feature}\'s feature is {model.feature_names_in_}')
