import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/lib')  # change to the path where the lib folder is located
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
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'  # where the data is saved
# trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/{target}/SHIP_Trend_test/{feature_comb}/'
# trained_models_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/XGBoost/{target}/{feature_comb}/'


df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test', 'Memory_Test'])

# %%
"""
Define feature lists
"""
# load the feature lists from feature_lists.pkl
feature_lists_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/feature_lists.pkl'  # where the feature lists are saved
with open(feature_lists_path, 'rb') as f:
    feature_lists = pickle.load(f)

Sleep = feature_lists[0]
Cov = feature_lists[1]
APOE4 = feature_lists[2]
TIV = feature_lists[3]
Thickness_DK = feature_lists[4]
Thickness_Schaefer = feature_lists[5]
Area_DK = feature_lists[6]
Area_Schaefer = feature_lists[7]
Subcortical = feature_lists[8]
targets = feature_lists[9]

sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']

"""
Data preprocessing for SHIP_Trend dataset
1. Convert sleep measurements units
2. Add group based on age and sex
3. Brain correction by brain size using internal data normalisation
4. NAI_Wordlist_Test transfer to accuracy
5. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_SHIP_ml = utils.convert_units(df_SHIP, sleep_dur_cols, sleep_eff_cols)

# add a column of 'Age_Group' after 'Age_at_Scan'.
df_SHIP_ml = utils.add_age_groups(df_SHIP_ml)
# add a column of Group which represents the group of age groups and SEX groups
df_SHIP_ml = utils.add_groups_age_sex(df_SHIP_ml)

# Brain correction by brain size using internal data normalisation
df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].div(df_SHIP_ml[Thickness_DK].sum(axis=1), axis=0)
df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].div(df_SHIP_ml[Thickness_Schaefer].sum(axis=1), axis=0)
df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].div(df_SHIP_ml[Area_DK].sum(axis=1), axis=0)
df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].div(df_SHIP_ml[Area_Schaefer].sum(axis=1), axis=0)
df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].div(df_SHIP_ml['EstimatedTotalIntraCranialVol'], axis=0)

# NAI_Wordlist_Test transfer to accuracy
df_SHIP_ml['Memory_Test'] = df_SHIP_ml['Memory_Test'].apply(lambda x: x / 16) * 100

# %%
# convert data types of columns. 'SEX' and 'APOE4' convert to object. Sleep, 'Age_at_Scan', 'BMI', TIV, and all imaging data to float64
columns_to_object = ['SEX', 'APOE4']
df_SHIP_ml[columns_to_object] = df_SHIP_ml[columns_to_object].astype('category')

# Convert specified columns to 'float64' data type
columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + targets
columns_to_float_KI = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + ['Memory_Test']
df_SHIP_ml[columns_to_float] = df_SHIP_ml[columns_to_float].astype('float64')

# %%
# feature_comb_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov', 'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
#                      'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle', 'Sleep_Cov_Brain_APOE',
#                      'Sleep_Cov_Brain_APOE_Shuffle']

feature_comb_list = ['Sleep_Cov']

for feature_comb in feature_comb_list:
    # Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
    if feature_comb == 'Sleep_Shuffle_Cov':
        df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
    if 'APOE_Shuffle' in feature_comb:
        df_SHIP_ml[APOE4] = df_SHIP_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
    if feature_comb == 'Sleep_Cov_Brain_Shuffle':
        df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
        df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
        df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
        df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
        df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

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
        'Sleep_Cov_Brain_APOE': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + APOE4,
        'Sleep_Cov_Brain_APOE_Shuffle': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + APOE4,
    }

    preprocessor_save_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/preprocessors_XGBoost_AutoGluon/{feature_comb}/'

    if not os.path.exists(preprocessor_save_path):
        os.makedirs(preprocessor_save_path)

    X = X_dict[feature_comb]
    df_train = df_SHIP_ml
    numeric_features = [feature for feature in X if df_SHIP_ml[feature].dtype.name in ['int64', 'float64']]

    """
    Make transformers for specific feature combinations and targets.
    After finished, this code snippet will be removed.
    """
    # Preprocessor definition
    whole_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            # ('cat', OneHotEncoder(), categorical_features)
        ], remainder='passthrough')
    whole_preprocessor.set_output(transform='pandas')

    whole_X_train_preprocessor = whole_preprocessor
    X_train = df_train[X]
    X_train = whole_X_train_preprocessor.fit_transform(X_train)

    # Save the transformer
    transformer_save_path = os.path.join(preprocessor_save_path, "whole_X_train_preprocessor.pkl")
    with open(transformer_save_path, 'wb') as f:
        pickle.dump(whole_X_train_preprocessor, f)  # if not usable, change to joblib.dump

    # # %%
    # # load the transformer
    # transformer_save_path = os.path.join(preprocessor_save_path, "whole_X_train_preprocessor.pkl")
    # with open(transformer_save_path, 'rb') as f:
    #     whole_X_train_preprocessor = pickle.load(f)  # if not usable, change to joblib.load
    #
    # # test whether the preprocessor's type is correct
    # print(type(whole_X_train_preprocessor))
