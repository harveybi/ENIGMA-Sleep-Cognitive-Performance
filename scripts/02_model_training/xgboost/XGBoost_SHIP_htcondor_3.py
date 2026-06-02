import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')

import utils

import argparse
import numpy as np
import pandas as pd
import pickle
import time
from joblib import dump, load, Parallel, delayed, parallel_config
from joblib_htcondor import register_htcondor
from datetime import datetime

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
import optuna

import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='XGBoost, SHIP_Trend and Liege dataset.')
parser.add_argument('feature_comb', type=str, help='''Name of the feature combination to be used.
    Available combinations are:
    - Sleep: Uses sleep features only.
    - Cov: Uses covariates only.
    - Sleep_Cov: Uses sleep features combined with covariates.
    - Sleep_Shuffle_Cov: Uses sleep features (shuffled) and covariates.
    - Brain: Uses brain features only.
    - Sleep_Cov_Brain: Uses sleep features, covariates, and brain features.
    - Sleep_Cov_Brain_Shuffle: Uses sleep features, covariates, and brain features (shuffled).
    - Sleep_APOE: Uses sleep features and APOE4.
    - Sleep_APOE_Shuffle: Uses sleep features (shuffled) and APOE4.
    - Sleep_Cov_APOE: Uses sleep features, covariates, and APOE4.
    - Sleep_Cov_APOE_Shuffle: Uses sleep features, covariates (shuffled), and APOE4.
    - Sleep_Cov_Brain_APOE: Uses sleep features, covariates, brain features, and APOE4.
    - Sleep_Cov_Brain_APOE_Shuffle: Uses sleep features, covariates, brain features (shuffled), and APOE4.
''')

parser.add_argument('target', type=str, help='''Name of the target to be predicted.
    Available targets are:
    - Stroop: Stroop_Test
    - Memory: Memory_Test
    - Stroop_rgo_age: Stroop_rgo_age
    - Memory_rgo_age: Memory_rgo_age
''')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target

print(f"\nStarting XGBoost pipeline for {target} prediction with feature combination {feature_comb}.\n")
start_time = datetime.now()

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/XGBoost/{target}/SHIP_Trend_htcondor/'

df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed.csv')
df_SHIP = df_SHIP.dropna(subset=['Stroop_Test', 'Memory_Test'])

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

# Shuffle the corresponding features when Sleep_Shuffle_Cov, Sleep_Cov_Subcor_Shuffle, Cov_Brain_Shuffle, Cov_Subcor_Shuffle
if feature_comb == 'Sleep_Shuffle_Cov':
    df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Subcor_Shuffle':
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Cov_Subcor':
    df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Shuffle_Subcor':
    df_SHIP_ml[Sleep] = df_SHIP_ml[Sleep].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Brain_Shuffle':
    df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Cov_Subcor_Shuffle':
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)
if 'APOE_Shuffle' in feature_comb:
    df_SHIP_ml[APOE4] = df_SHIP_ml[APOE4].sample(frac=1, random_state=33).reset_index(drop=True)
if feature_comb == 'Sleep_Cov_Brain_Shuffle':
    df_SHIP_ml[Thickness_DK] = df_SHIP_ml[Thickness_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Thickness_Schaefer] = df_SHIP_ml[Thickness_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_DK] = df_SHIP_ml[Area_DK].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Area_Schaefer] = df_SHIP_ml[Area_Schaefer].sample(frac=1, random_state=33).reset_index(drop=True)
    df_SHIP_ml[Subcortical] = df_SHIP_ml[Subcortical].sample(frac=1, random_state=33).reset_index(drop=True)

# %%
"""
Pipeline
"""
class AdMLPipeline:
    def __init__(self, X, y, df_train=None, num_cores=None, save_path=None, model=None):
        self.X = X
        self.y = y
        self.df_train = df_train
        self.num_cores = num_cores
        self.save_path = save_path
        self.model = model
        self.rgo_age = False
        self.whole_X_train_preprocessor = None
        self.whole_age_poly = None
        self.whole_age_lr = None

    def generate_kfold_finalized(self, y=None, n_splits=5, random_state=0, stratified=False, n_repeats=1):
        output = []
        X_data = self.df_train
        y_data = self.df_train[y] if y is not None else None

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

    def xgboost_objective(self, trial, X_train, y_train):
        param = {
            'tree_method': 'hist',
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 100.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 100.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 25, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 40),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_jobs': 1
        }

        model = xgb.XGBRegressor(**param, random_state=42, enable_categorical=True)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # cross_val_score runs with n_jobs=-1, using the nested backend
        print(f"[Optuna Trial {trial.number}] Launching nested parallel cross_val_score...")
        start_cv_time = time.time()
        scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=cv, n_jobs=-1)
        print(f"[Optuna Trial {trial.number}] Nested cross_val_score finished in {time.time() - start_cv_time:.2f} sec. Result: {np.mean(scores):.4f}")

        return np.mean(scores)

    def xgboost_cross_val_iteration(self, repeat_fold, train_index, test_index):
        print(f"\nRepeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1} start")
        # print(f"Featurs:{self.X}, target:{self.y}\n")
        # Extract data as DataFrame instead of numpy arrays to preserve column names
        X_train = self.df_train.iloc[train_index][self.X]
        y_train = self.df_train.iloc[train_index][self.y]
        X_test = self.df_train.iloc[test_index][self.X]
        y_test = self.df_train.iloc[test_index][self.y]

        # Identify categorical and numeric features based on dtype
        # categorical_features = [feature for feature in self.X if self.df_train[feature].dtype.name == 'category']
        numeric_features = [feature for feature in self.X if self.df_train[feature].dtype.name in ['int64', 'float64']]

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

        if self.rgo_age:
            age_train = self.df_train.iloc[train_index]['Age_at_Scan'].values.reshape(-1, 1)
            age_test = self.df_train.iloc[test_index]['Age_at_Scan'].values.reshape(-1, 1)

            # Create polynomial features
            poly = PolynomialFeatures(degree=2)  # You can adjust the degree as necessary
            age_train_poly = poly.fit_transform(age_train)
            age_test_poly = poly.transform(age_test)

            # Fit a linear model to remove age effect
            lr = LinearRegression()
            lr.fit(age_train_poly, y_train)

            # print the r2 score of the linear regression model on train and test data
            print(f"\nLinear regression model r2 score on train data: {lr.score(age_train_poly, y_train)}")
            print(f"Linear regression model r2 score on test data: {lr.score(age_test_poly, y_test)}")

            # Predict and remove age-related effects from the target
            y_train = y_train - lr.predict(age_train_poly)
            y_test = y_test - lr.predict(age_test_poly)


        print(f"[{repeat_fold}] Starting sequential hyperparameter optimization (Optuna trials)...")
        optuna_start = time.time()
        sampler = optuna.samplers.TPESampler(seed=42)
        study_name = f"study_repeat{repeat_fold[0]}_fold{repeat_fold[1]}"
        study = optuna.create_study(direction='maximize', sampler=sampler, study_name=study_name)

        try:
            # Optuna runs trials sequentially (n_jobs=1)
            study.optimize(lambda trial: self.xgboost_objective(trial, X_train, y_train), n_trials=150, n_jobs=1)
            best_params = study.best_params
            best_value = study.best_value
            print(
                f"[{repeat_fold}] Optuna finished (sequential trials) in {time.time() - optuna_start:.2f} sec. Best Obj: {best_value:.4f}")
            print(f"[{repeat_fold}] Best Params: {best_params}")
        except Exception as e:
            print(f"Error during Optuna optimization in fold {repeat_fold}: {e}")
            return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': {}, 'test_r2': -np.inf,
                    'error': f"Optuna failed: {e}"}

        print(f"[{repeat_fold}] Training final model for the fold...")
        fit_start = time.time()
        final_model = xgb.XGBRegressor(**best_params, random_state=42, enable_categorical=True, n_jobs=1)  # n_jobs=1
        final_model.fit(X_train, y_train)
        print(f"[{repeat_fold}] Fold model fitting finished in {time.time() - fit_start:.2f} sec.")


        preds = final_model.predict(X_test)
        # print mean of y_train, y_test, and pred
        print(f"y_train mean: {y_train.mean()}")
        print(f"y_test mean: {y_test.mean()}")
        print(f"pred mean: {preds.mean()}")
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        corr = np.corrcoef(y_test, preds)[0, 1]
        corr_spearman, _ = spearmanr(y_test, preds)

        print(f"\nRepeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1} end\n")

        return {
            'repeat': repeat_fold[0] + 1,
            'fold': repeat_fold[1] + 1,
            'params': best_params,
            'test_r2': r2,
            'test_neg_root_mean_squared_error': rmse,
            'test_neg_mean_absolute_error': mae,
            'test_r_corr': corr,
            'test_spearmanr': corr_spearman
        }

    def xgboost_pipe(self, stratified_label=None):
        if stratified_label is not None:
            cv_splitter = self.generate_kfold_finalized(y=stratified_label, n_splits=5, random_state=42,
                                                        stratified=True, n_repeats=1)  # 5, 10
        else:
            cv_splitter = self.generate_kfold_finalized(y=None, n_splits=5, random_state=42, stratified=False,
                                                        n_repeats=1)  # 5, 10

        if 'Age_at_Scan' in self.X and 'rgo_age' in self.y:
            if 'Stroop_Test' in self.y:
                self.y = 'Stroop_Test'
            elif 'Memory_Test' in self.y:
                self.y = 'Memory_Test'

            self.rgo_age = True
            self.X.remove('Age_at_Scan')

        num_jobs = len(cv_splitter)
        scores_list = []
        total_start_time = time.time()
        print("Starting parallel execution (Outer CV Folds)...")

        # Configure parallel_config for OUTER loop parallelism ONLY
        # max_recursion_level=0 disables nesting
        # throttle is not strictly needed here but kept for consistency
        register_htcondor("INFO")
        with parallel_config(
                backend="htcondor",
                pool="head2.htc.inm7.de",
                request_cpus=1,
                request_disk="1GB",
                request_memory="2Gb",
                throttle=[num_jobs, 50],  # Throttle for the outer jobs only
                export_metadata=True,
                shared_data_dir='/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/XGBoost/joblib_htcondor',
                max_recursion_level=1,  # IMPORTANT: Disable nesting
                n_jobs=-1,
        ):
            print(f"Submitting {num_jobs} outer CV jobs...")
            scores_list = Parallel()(
                delayed(self.xgboost_cross_val_iteration)(repeat_fold, train_index, test_index)
                for repeat_fold, train_index, test_index in cv_splitter
            )

        total_end_time = time.time()
        print(f"\nParallel execution finished in {total_end_time - total_start_time:.2f} seconds.")

        valid_scores = [s for s in scores_list if s and 'error' not in s]
        error_scores = [s for s in scores_list if not s or 'error' in s]
        if not valid_scores: print("ERROR: All folds failed."); return pd.DataFrame(), None
        if error_scores: print(f"Warning: {len(error_scores)} folds failed.")

        scores_df = pd.DataFrame(valid_scores)
        scores_df.sort_values(by='test_r2', ascending=False, inplace=True)
        if scores_df.empty: print("ERROR: No valid scores."); return pd.DataFrame(), None

        overall_best_params = scores_df.iloc[0]['params']
        print("\nCV results summary:")
        print(scores_df[['repeat', 'fold', 'test_r2']].describe())
        print(f"\nMean R2: {scores_df['test_r2'].mean():.4f}")
        print(f"\nOverall best parameters: {overall_best_params}")

        print("Refitting final model sequentially...")
        refit_start = time.time()
        numeric_features = [f for f in self.X if self.df_train[f].dtype.name in ['int64', 'float64']]
        self.whole_X_train_preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)],
                                                            remainder='passthrough')
        self.whole_X_train_preprocessor.set_output(transform='pandas')
        X_train_full = self.df_train[self.X]
        y_train_full = self.df_train[self.y].copy()
        X_train_full_processed = self.whole_X_train_preprocessor.fit_transform(X_train_full)
        X_train_full_processed.columns = [c.split('__')[-1] for c in X_train_full_processed.columns]

        if self.rgo_age:
            print("Applying age regression to full dataset...")
            age_train_full = self.df_train['Age_at_Scan'].values.reshape(-1, 1)
            self.whole_age_poly = PolynomialFeatures(degree=2)
            age_train_full_poly = self.whole_age_poly.fit_transform(age_train_full)
            self.whole_age_lr = LinearRegression();
            self.whole_age_lr.fit(age_train_full_poly, y_train_full)
            print(f"  Full data Age Regressor R2: {self.whole_age_lr.score(age_train_full_poly, y_train_full):.4f}")
            y_train_full_final = y_train_full - self.whole_age_lr.predict(age_train_full_poly)
        else:
            y_train_full_final = y_train_full

        final_best_model = xgb.XGBRegressor(**overall_best_params, random_state=42, enable_categorical=True,
                                            n_jobs=-1)  # Use local cores
        final_best_model.fit(X_train_full_processed, y_train_full_final)
        print(f"Final model refitting finished in {time.time() - refit_start:.2f} seconds.")

        self.model = final_best_model

        return scores_df, self.model

# %%
"""
Define X and y
"""
X_dict = {
    'Sleep': Sleep,
    'Cov': Cov,
    'Brain': Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'CT': Thickness_DK + Thickness_Schaefer,
    'SA': Area_DK + Area_Schaefer,
    'Subcor': Subcortical,
    'Sleep_Cov': Sleep + Cov,
    'Sleep_Cov_Brain': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_Cov_CT': Sleep + Cov + Thickness_DK + Thickness_Schaefer,
    'Sleep_Cov_SA': Sleep + Cov + Area_DK + Area_Schaefer,
    'Sleep_Cov_Subcor': Sleep + Cov + Subcortical,
    'Sleep_Brain': Sleep + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_CT': Sleep + Thickness_DK + Thickness_Schaefer,
    'Sleep_SA': Sleep + Area_DK + Area_Schaefer,
    'Sleep_Subcor': Sleep + Subcortical,
    'Cov_Brain': Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Cov_CT': Cov + Thickness_DK + Thickness_Schaefer,
    'Cov_SA': Cov + Area_DK + Area_Schaefer,
    'Cov_Subcor': Cov + Subcortical,
    'Sleep_Shuffle_Cov': Sleep + Cov,
    'Sleep_Cov_Subcor_Shuffle': Sleep + Cov + Subcortical,
    'Sleep_Shuffle_Cov_Subcor': Sleep + Cov + Subcortical,
    'Sleep_Shuffle_Subcor': Sleep + Subcortical,
    'Cov_Brain_Shuffle': Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Cov_Subcor_Shuffle': Cov + Subcortical,
    'Sleep_APOE': Sleep + APOE4,
    'Sleep_APOE_Shuffle': Sleep + APOE4,
    'Sleep_Cov_APOE': Sleep + Cov + APOE4,
    'Sleep_Cov_APOE_Shuffle': Sleep + Cov + APOE4,
    'Cov_APOE': Cov + APOE4,
    'Cov_APOE_Shuffle': Cov + APOE4,
    'Brain_APOE': Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + APOE4,
    'Brain_APOE_Shuffle': Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + APOE4,
    'Subcor_APOE': Subcortical + APOE4,
    'Subcor_APOE_Shuffle': Subcortical + APOE4,
    'Sleep_Cov_Brain_APOE': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + APOE4,
    'Sleep_Cov_Brain_APOE_Shuffle': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + APOE4,
    'Sleep_Cov_Subcor_APOE': Sleep + Cov + Subcortical + APOE4,
    'Sleep_Cov_Subcor_APOE_Shuffle': Sleep + Cov + Subcortical + APOE4,
    'Sleep_Cov_Brain_Shuffle': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical
}

y_dict = {
    'Stroop': 'Stroop_Test',
    'Memory': 'Memory_Test',
    'Stroop_rgo_age': 'Stroop_Test_rgo_age',
    'Memory_rgo_age': 'Memory_Test_rgo_age'
}

# %%
print(f"\nStart model training")
train_start_time = datetime.now()

case_results_path = results_path + f"{feature_comb}/"
if not os.path.exists(case_results_path):
    os.makedirs(case_results_path)

columns_to_int = ['SEX', 'APOE4']
df_SHIP_ml[columns_to_int] = df_SHIP_ml[columns_to_int].fillna(0).astype('int64')
columns_to_cat = ['SEX', 'APOE4']
df_SHIP_ml[columns_to_cat] = df_SHIP_ml[columns_to_cat].astype('category')

cat_dtypes = {col: df_SHIP_ml[col].dtype for col in columns_to_cat}

with open(case_results_path + "cat_dtypes.pkl", "wb") as f:
    pickle.dump(cat_dtypes, f)

columns_to_float = Sleep + ['Age_at_Scan', 'BMI', TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + targets
df_SHIP_ml[columns_to_float] = df_SHIP_ml[columns_to_float].astype('float64')

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_SHIP_ml

# %%
ml_pipe_instance = AdMLPipeline(X, y, df_train=df_train, save_path=case_results_path)
scores, model = ml_pipe_instance.xgboost_pipe(stratified_label='Group_Age_SEX')

# calculate used time
train_end_time = datetime.now()

# time format: hours, minutes, seconds
print(f"\nModel training completed. Time used: {train_end_time - train_start_time}")

# %%
# save score and model
scores.to_csv(case_results_path + 'scores.csv')

model_save_path = os.path.join(case_results_path, "xgboost_model.json")
# Save the trained XGBoost model
model.save_model(model_save_path)

# %%
print(f"\nDone! XGBoost pipeline for {target} prediction by {feature_comb} completed successfully.")

# print Date and Time
print("Current date and time: ", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
# print total execution time by hours and minutes
execution_time = datetime.now() - start_time
print(f"Execution Time: {execution_time}\n")
