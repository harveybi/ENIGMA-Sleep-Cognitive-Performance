import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')

import utils
import XGBoost_pipeline
from XGBoost_pipeline import AdMLPipeline
# import advanced_ml_pipeline
# from advanced_ml_pipeline import AdMLPipeline

import argparse
import pandas as pd
import shap
import pickle
import time
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score

from dask.distributed import Client
from dask_jobqueue.htcondor import HTCondorCluster

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'

df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed.csv')

# for df_SHIP, df_Liege, df_Liege_COF, df_Liege_COGNAP, remove subjects with missing values in 'Stroop_Test' and 'Memory_Test'
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test', 'Memory_Test'])
df_Liege = df_Liege.dropna(subset=['Stroop_Test', 'Memory_Test'])

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
2. Add group based on age and sex
3. Brain correction by brain size using internal data normalisation
4. NAI_Wordlist_Test transfer to accuracy
5. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']
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
"""
Data preprocessing for Liege dataset
1. Convert sleep measurements units
2. Brain correction by brain size using internal data normalisation
3. Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
"""
# convert sleep measurements units
df_Liege_ml = utils.convert_units(df_Liege, sleep_dur_cols, sleep_eff_cols)

# Brain correction by brain size using internal data normalisation
df_Liege_ml[Thickness_DK] = df_Liege_ml[Thickness_DK].div(df_Liege_ml[Thickness_DK].sum(axis=1), axis=0)
df_Liege_ml[Thickness_Schaefer] = df_Liege_ml[Thickness_Schaefer].div(df_Liege_ml[Thickness_Schaefer].sum(axis=1),
                                                                      axis=0)
df_Liege_ml[Area_DK] = df_Liege_ml[Area_DK].div(df_Liege_ml[Area_DK].sum(axis=1), axis=0)
df_Liege_ml[Area_Schaefer] = df_Liege_ml[Area_Schaefer].div(df_Liege_ml[Area_Schaefer].sum(axis=1), axis=0)
df_Liege_ml[Subcortical] = df_Liege_ml[Subcortical].div(df_Liege_ml['EstimatedTotalIntraCranialVol'], axis=0)

df_Liege_ml['Memory_Test'] = df_Liege_ml['Memory_Test'] * 100
df_Liege_ml['1-back_acc'] = df_Liege_ml['1-back_acc'] * 100
df_Liege_ml['2-back_acc'] = df_Liege_ml['2-back_acc'] * 100
df_Liege_ml['3-back_acc'] = df_Liege_ml['3-back_acc'] * 100

columns_to_object = ['SEX', 'APOE4']
df_SHIP_ml[columns_to_object] = df_SHIP_ml[columns_to_object].astype('category')
df_Liege_ml[columns_to_object] = df_Liege_ml[columns_to_object].astype('category')

# %%
features = Sleep + Cov

X_train = df_SHIP_ml[features]
X_test = df_Liege_ml[features]

# numeric_features = ['PSG_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Dur', 'Self_Sleep_Eff', 'Depression_score', 'Age_at_Scan', 'BMI']
numeric_features = [feature for feature in features if X_train[feature].dtype.name in ['int64', 'float64']]

# %%
preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features)
            ], remainder='passthrough')
preprocessor.set_output(transform='pandas')

# Apply transformations
print(f"Type of preprocessor before fitting: {type(preprocessor)}")
print(f"Type of X_train before fitting: {type(X_train)}")
try:
    X_train = preprocessor.fit_transform(X_train)
    print("X_train transformed successfully")
    # for column names in X_train, remove __ and before __
    X_train.columns = [column.split('__')[-1] for column in X_train.columns]
    # print(f"Type of preprocessor after fitting: {type(preprocessor)}")
    # print(f"Type of X_train after fitting: {type(X_train)}")
except Exception as e:
    print(f"Error during transforming X_train: {e}")
    print(f"Type of preprocessor after fitting: {type(preprocessor)}")

print(f"Type of X_test before transforming: {type(X_test)}")
try:
    X_test = preprocessor.transform(X_test)
    print("X_test transformed successfully")
    # for column names in X_test, remove __ and before __
    X_test.columns = [column.split('__')[-1] for column in X_test.columns]
    # print(f"Type of X_test after transforming: {type(X_test)}")
except Exception as e:
    print(f"Error during transforming X_test: {e}")
    print(f"Type of preprocessor for transform: {type(preprocessor)}")
    print(f"Type of X_test after transforming: {type(X_test)}")

# %%
# make a 5 times loop for the processor pipeline
for i in range(5):
    print(f"Loop {i + 1}")
    # create a new preprocessor for each loop
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ], remainder='passthrough')
    preprocessor.set_output(transform='pandas')

    # Apply transformations
    print(f"Type of preprocessor before fitting: {type(preprocessor)}")
    print(f"Type of X_train before fitting: {type(X_train)}")
    try:
        X_train = preprocessor.fit_transform(X_train)
        print("X_train transformed successfully")
        # for column names in X_train, remove __ and before __
        X_train.columns = [column.split('__')[-1] for column in X_train.columns]
        # print(f"Type of preprocessor after fitting: {type(preprocessor)}")
        # print(f"Type of X_train after fitting: {type(X_train)}")
    except Exception as e:
        print(f"Error during transforming X_train: {e}")
        print(f"Type of preprocessor after fitting: {type(preprocessor)}")

    print(f"Type of X_test before transforming: {type(X_test)}")
    try:
        X_test = preprocessor.transform(X_test)
        print("X_test transformed successfully")
        # for column names in X_test, remove __ and before __
        X_test.columns = [column.split('__')[-1] for column in X_test.columns]
        # print(f"Type of X_test after transforming: {type(X_test)}")
    except Exception as e:
        print(f"Error during transforming X_test: {e}")
        print(f"Type of preprocessor for transform: {type(preprocessor)}")
        print(f"Type of X_test after transforming: {type(X_test)}")


# %%
def xgboost_cross_val_iteration(repeat_fold, train_index, test_index):
    print(f"\nRepeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1} start")
    # print(f"Featurs:{self.X}, target:{self.y}\n")
    # Extract data as DataFrame instead of numpy arrays to preserve column names
    X_train = df_train.iloc[train_index][X]
    y_train = df_train.iloc[train_index][y]
    X_test = df_train.iloc[test_index][X]
    y_test = df_train.iloc[test_index][y]

    # Identify categorical and numeric features based on dtype
    # categorical_features = [feature for feature in self.X if self.df_train[feature].dtype.name == 'category']
    numeric_features = [feature for feature in X if df_train[feature].dtype.name in ['int64', 'float64']]

    # Define preprocessing for numeric and categorical data using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ], remainder='passthrough')
    preprocessor.set_output(transform='pandas')

    # Apply transformations
    print(f"Type of preprocessor before fitting: {type(preprocessor)}")
    print(f"Type of X_train before fitting: {type(X_train)}")
    try:
        X_train = preprocessor.fit_transform(X_train)
        print("X_train transformed successfully")
        X_train.columns = [column.split('__')[-1] for column in X_train.columns]
        # print(f"Type of preprocessor after fitting: {type(preprocessor)}")
        # print(f"Type of X_train after fitting: {type(X_train)}")
    except Exception as e:
        print(f"Error during transforming X_train: {e}")
        print(f"Type of preprocessor after fitting: {type(preprocessor)}")

    print(f"Type of X_test before transforming: {type(X_test)}")
    try:
        X_test = preprocessor.transform(X_test)
        print("X_test transformed successfully")
        X_test.columns = [column.split('__')[-1] for column in X_test.columns]
        # print(f"Type of X_test after transforming: {type(X_test)}")
    except Exception as e:
        print(f"Error during transforming X_test: {e}")
        print(f"Type of preprocessor for transform: {type(preprocessor)}")
        print(f"Type of X_test after transforming: {type(X_test)}")


# %%
df_train = df_SHIP_ml
X = features
y = 'Stroop_Test'

def generate_kfold_finalized(y=None, n_splits=5, random_state=0, stratified=False, n_repeats=1):
    output = []
    X_data = df_train
    y_data = df_train[y] if y is not None else None

    if stratified and (y is not None):
        if n_repeats > 1:
            kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for idx, (train_index, test_index) in enumerate(kf.split(X_data, y_data)):
            repeat = idx // n_splits
            fold = idx % n_splits
            output.append(((repeat, fold), train_index, test_index))

    else:
        if n_repeats > 1:
            kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for idx, (train_index, test_index) in enumerate(kf.split(X_data)):
            repeat = idx // n_splits
            fold = idx % n_splits
            output.append(((repeat, fold), train_index, test_index))

    return output


cv_splitter = generate_kfold_finalized(y=None, n_splits=5, random_state=42,
                                       stratified=True, n_repeats=1)  # 5, 10

for repeat_fold, train_index, test_index in cv_splitter:
    xgboost_cross_val_iteration(repeat_fold, train_index, test_index)

