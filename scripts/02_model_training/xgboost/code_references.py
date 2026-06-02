"""
Parallel Outer Loop Only
Pipeline with joblib-htcondor integration supporting nested parallelism.
Configuration:
- Outer Loop (Cross-Validation Folds): Parallel via HTCondor
- Middle Loop (Optuna Trials): Sequential within each Outer Job
- Inner Loop (cross_val_score within Optuna Objective): Sequential within each Outer Job
"""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
)
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import os

# --- Joblib and HTCondor Imports ---
from joblib import Parallel, delayed, parallel_config, __version__ as joblib_version
try:
    from joblib_htcondor import register_htcondor
    register_htcondor()
    HTCONDOR_AVAILABLE = True
    print(f"joblib-htcondor backend registered successfully (joblib version: {joblib_version}).")
except ImportError:
    HTCONDOR_AVAILABLE = False
    print("Warning: joblib-htcondor not found. Parallelism requires this backend.")
    print("Install it using: pip install joblib-htcondor")
# ------------------------------------

class AdMLPipeline:
    def __init__(self, X, y, df_train=None, save_path=None, model=None, shared_data_dir=None):
        """
        Initializes the pipeline.

        Args:
            X (list): List of feature column names.
            y (str): Target column name.
            df_train (pd.DataFrame, optional): Training dataframe. Defaults to None.
            save_path (str, optional): Path to save results/models. Defaults to None.
            model (object, optional): Pre-trained model. Defaults to None.
            shared_data_dir (str, optional): Path to a directory accessible by all HTCondor nodes
                                             for joblib data sharing. Required for HTCondor backend.
                                             Defaults to None.
        """
        self.X = X
        self.y = y
        self.df_train = df_train
        self.save_path = save_path
        self.model = model
        self.rgo_age = False
        self.whole_X_train_preprocessor = None
        self.whole_age_poly = None
        self.whole_age_lr = None
        if shared_data_dir and not os.path.exists(shared_data_dir):
             try:
                 os.makedirs(shared_data_dir)
                 print(f"Created shared data directory: {shared_data_dir}")
             except OSError as e:
                 print(f"Warning: Could not create shared data directory {shared_data_dir}: {e}")
        self.shared_data_dir = shared_data_dir

    def generate_kfold_finalized(self, y=None, n_splits=5, random_state=0, stratified=False, n_repeats=1):
        """Generates K-Fold or Stratified K-Fold indices."""
        output = []
        X_data = self.df_train
        y_data = self.df_train[y] if y is not None else None

        if stratified and (y is not None):
            if n_repeats > 1: kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else: kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for idx, (train_index, test_index) in enumerate(kf.split(X_data, y_data)):
                output.append(((idx // n_splits, idx % n_splits), train_index, test_index))
        else:
            if n_repeats > 1: kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else: kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for idx, (train_index, test_index) in enumerate(kf.split(X_data)):
                output.append(((idx // n_splits, idx % n_splits), train_index, test_index))
        return output

    def xgboost_objective(self, trial, X_train, y_train):
        """
        Optuna objective function. Runs SEQUENTIALLY within the outer job.
        cross_val_score also runs SEQUENTIALLY.
        """
        param = {
            'tree_method': 'hist', 'lambda': trial.suggest_loguniform('lambda', 1e-3, 100.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 100.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': 1 # XGBoost runs sequentially
        }
        model = xgb.XGBRegressor(**param, random_state=42, enable_categorical=True)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        # cross_val_score runs sequentially (n_jobs=1)
        scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=cv, n_jobs=1)
        return np.mean(scores)

    def xgboost_cross_val_iteration(self, repeat_fold, train_index, test_index):
        """
        Performs a single cross-validation iteration. Runs as a PARALLEL outer job.
        Optuna trials run sequentially within this job.
        cross_val_score within Optuna objective runs sequentially.
        """
        print(f"\nStarting Outer Job - Repeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1}")
        start_time = time.time()
        X_train_df = self.df_train.iloc[train_index][self.X]
        y_train = self.df_train.iloc[train_index][self.y]
        X_test_df = self.df_train.iloc[test_index][self.X]
        y_test = self.df_train.iloc[test_index][self.y]
        numeric_features = [f for f in self.X if self.df_train[f].dtype.name in ['int64', 'float64']]
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')
        preprocessor.set_output(transform='pandas')

        try:
            X_train = preprocessor.fit_transform(X_train_df); X_train.columns = [c.split('__')[-1] for c in X_train.columns]
            X_test = preprocessor.transform(X_test_df); X_test.columns = [c.split('__')[-1] for c in X_test.columns]
            print(f"[{repeat_fold}] Preprocessing successful.")
        except Exception as e:
            print(f"Error during preprocessing in fold {repeat_fold}: {e}")
            return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': {}, 'test_r2': -np.inf, 'error': str(e)} # Simplified error return

        if self.rgo_age:
            print(f"[{repeat_fold}] Regressing out age effect...")
            age_train = self.df_train.iloc[train_index]['Age_at_Scan'].values.reshape(-1, 1)
            age_test = self.df_train.iloc[test_index]['Age_at_Scan'].values.reshape(-1, 1)
            poly = PolynomialFeatures(degree=2); age_train_poly = poly.fit_transform(age_train); age_test_poly = poly.transform(age_test)
            lr = LinearRegression(); lr.fit(age_train_poly, y_train)
            print(f"[{repeat_fold}]   Age Regressor R2 (Train): {lr.score(age_train_poly, y_train):.4f}")
            print(f"[{repeat_fold}]   Age Regressor R2 (Test): {lr.score(age_test_poly, y_test):.4f}")
            y_train = y_train - lr.predict(age_train_poly); y_test = y_test - lr.predict(age_test_poly)

        print(f"[{repeat_fold}] Starting sequential hyperparameter optimization (Optuna trials)...")
        optuna_start = time.time()
        sampler = optuna.samplers.TPESampler(seed=42)
        study_name = f"study_repeat{repeat_fold[0]}_fold{repeat_fold[1]}"
        study = optuna.create_study(direction='maximize', sampler=sampler, study_name=study_name)

        try:
            # Optuna runs trials sequentially (n_jobs=1)
            study.optimize(lambda trial: self.xgboost_objective(trial, X_train, y_train), n_trials=150, n_jobs=1)
            best_params = study.best_params; best_value = study.best_value
            print(f"[{repeat_fold}] Optuna finished (sequential trials) in {time.time() - optuna_start:.2f} sec. Best Obj: {best_value:.4f}")
            print(f"[{repeat_fold}] Best Params: {best_params}")
        except Exception as e:
             print(f"Error during Optuna optimization in fold {repeat_fold}: {e}")
             return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': {}, 'test_r2': -np.inf, 'error': f"Optuna failed: {e}"}

        print(f"[{repeat_fold}] Training final model for the fold...")
        fit_start = time.time()
        final_model = xgb.XGBRegressor(**best_params, random_state=42, enable_categorical=True, n_jobs=1) # n_jobs=1
        final_model.fit(X_train, y_train)
        print(f"[{repeat_fold}] Fold model fitting finished in {time.time() - fit_start:.2f} sec.")

        preds = final_model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        try: corr = np.corrcoef(y_test, preds)[0, 1] if len(np.unique(y_test)) > 1 and len(np.unique(preds)) > 1 else np.nan
        except ValueError: corr = np.nan
        try: corr_spearman, _ = spearmanr(y_test, preds) if len(np.unique(y_test)) > 1 and len(np.unique(preds)) > 1 else (np.nan, np.nan)
        except ValueError: corr_spearman = np.nan
        print(f"[{repeat_fold}] Metrics: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Corr={corr:.4f}, Spearman={corr_spearman:.4f}")

        fold_time = time.time() - start_time
        print(f"Finished Outer Job - Repeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1} in {fold_time:.2f} seconds\n")
        return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': best_params, 'test_r2': r2,
                'test_neg_root_mean_squared_error': rmse, 'test_neg_mean_absolute_error': mae,
                'test_r_corr': corr, 'test_spearmanr': corr_spearman, 'fold_time_seconds': fold_time}

    def xgboost_pipe(self, stratified_label=None, n_repeats=10, n_splits=5):
        """
        Main pipeline: Outer CV Folds run in PARALLEL. Inner loops run sequentially.
        """
        if not HTCONDOR_AVAILABLE:
             print("ERROR: HTCondor backend not available.")
             return pd.DataFrame(), None
        elif self.shared_data_dir is None:
            print("ERROR: `shared_data_dir` must be specified.")
            return pd.DataFrame(), None
        else:
            backend_to_use = 'htcondor'
            print(f"Using HTCondor backend (Outer Parallel Only). Shared data: {self.shared_data_dir}")

        print(f"Generating {n_repeats} repeats of {n_splits}-fold CV splits...")
        stratified = stratified_label is not None
        cv_splitter = self.generate_kfold_finalized(y=stratified_label, n_splits=n_splits, random_state=42, stratified=stratified, n_repeats=n_repeats)
        num_jobs = len(cv_splitter)
        print(f"Total outer jobs (CV folds) to submit: {num_jobs}")

        if isinstance(self.y, str) and 'rgo_age' in self.y:
            print("Age regression enabled...")
            original_y = self.y
            if 'Stroop_Test' in self.y: self.y = 'Stroop_Test'
            elif 'Memory_Test' in self.y: self.y = 'Memory_Test'
            else:
                 base_y = self.y.replace('_rgo_age', ''); self.y = base_y if base_y in self.df_train.columns else original_y
            self.rgo_age = True
            if 'Age_at_Scan' in self.X: self.X = [f for f in self.X if f != 'Age_at_Scan']
            print(f"Target: {self.y}, Features: {self.X}")

        scores_list = []
        total_start_time = time.time()
        print("Starting parallel execution (Outer CV Folds)...")
        log_dir = os.path.join(self.shared_data_dir, f"joblib_logs_{self.y}_outer")
        if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

        # Configure parallel_config for OUTER loop parallelism ONLY
        # max_recursion_level=0 disables nesting
        # throttle is not strictly needed here but kept for consistency
        with parallel_config(
            backend="htcondor",
            pool="head2.htc.inm7.de", request_cpus=1, request_disk="1GB", request_memory="2Gb",
            throttle=25, # Throttle for the outer jobs only
            export_metadata=True,
            shared_data_dir=self.shared_data_dir, log_dir_prefix=log_dir,
            max_recursion_level=0,   # IMPORTANT: Disable nesting
            n_jobs=-1, pre_dispatch='all',
        ) as config:
            print(f"Submitting {num_jobs} outer CV jobs (NO nesting, throttle={config.get('throttle')})...")
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

        scores_df = pd.DataFrame(valid_scores); scores_df.sort_values(by='test_r2', ascending=False, inplace=True)
        if scores_df.empty: print("ERROR: No valid scores."); return pd.DataFrame(), None

        overall_best_params = scores_df.iloc[0]['params']
        print("\nCV results summary:"); print(scores_df[['repeat', 'fold', 'test_r2']].describe())
        print(f"\nOverall best parameters: {overall_best_params}")

        print("Refitting final model sequentially...")
        refit_start = time.time()
        numeric_features = [f for f in self.X if self.df_train[f].dtype.name in ['int64', 'float64']]
        self.whole_X_train_preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')
        self.whole_X_train_preprocessor.set_output(transform='pandas')
        X_train_full = self.df_train[self.X]; y_train_full = self.df_train[self.y].copy()
        X_train_full_processed = self.whole_X_train_preprocessor.fit_transform(X_train_full)
        X_train_full_processed.columns = [c.split('__')[-1] for c in X_train_full_processed.columns]

        if self.rgo_age:
            print("Applying age regression to full dataset...")
            age_train_full = self.df_train['Age_at_Scan'].values.reshape(-1, 1)
            self.whole_age_poly = PolynomialFeatures(degree=2); age_train_full_poly = self.whole_age_poly.fit_transform(age_train_full)
            self.whole_age_lr = LinearRegression(); self.whole_age_lr.fit(age_train_full_poly, y_train_full)
            print(f"  Full data Age Regressor R2: {self.whole_age_lr.score(age_train_full_poly, y_train_full):.4f}")
            y_train_full_final = y_train_full - self.whole_age_lr.predict(age_train_full_poly)
        else: y_train_full_final = y_train_full

        final_best_model = xgb.XGBRegressor(**overall_best_params, random_state=42, enable_categorical=True, n_jobs=-1) # Use local cores
        final_best_model.fit(X_train_full_processed, y_train_full_final)
        print(f"Final model refitting finished in {time.time() - refit_start:.2f} seconds.")

        self.model = final_best_model
        if self.save_path:
            try:
                os.makedirs(self.save_path, exist_ok=True)
                scores_filename = os.path.join(self.save_path, f"cv_scores_{self.y}_outer_parallel.csv")
                scores_df.to_csv(scores_filename, index=False)
                print(f"CV scores saved to {scores_filename}")
            except Exception as e: print(f"Warning: Could not save results: {e}")
        return scores_df, self.model

# %%
"""
Parallel Inner Loop Only (cross_val_score)
Pipeline with joblib-htcondor integration supporting nested parallelism.
Configuration:
- Outer Loop (Cross-Validation Folds): Sequential
- Middle Loop (Optuna Trials): Sequential within each Outer Step
- Inner Loop (cross_val_score within Optuna Objective): Parallel via HTCondor (Nested)
"""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
)
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import os

# --- Joblib and HTCondor Imports ---
from joblib import Parallel, delayed, parallel_config, __version__ as joblib_version
try:
    from joblib_htcondor import register_htcondor
    register_htcondor()
    HTCONDOR_AVAILABLE = True
    print(f"joblib-htcondor backend registered successfully (joblib version: {joblib_version}).")
except ImportError:
    HTCONDOR_AVAILABLE = False
    print("Warning: joblib-htcondor not found. Nested parallelism requires this backend.")
    print("Install it using: pip install joblib-htcondor")
# ------------------------------------

class AdMLPipeline:
    def __init__(self, X, y, df_train=None, save_path=None, model=None, shared_data_dir=None):
        """
        Initializes the pipeline.

        Args:
            X (list): List of feature column names.
            y (str): Target column name.
            df_train (pd.DataFrame, optional): Training dataframe. Defaults to None.
            save_path (str, optional): Path to save results/models. Defaults to None.
            model (object, optional): Pre-trained model. Defaults to None.
            shared_data_dir (str, optional): Path to a directory accessible by all HTCondor nodes
                                             for joblib data sharing. Required for HTCondor backend.
                                             Defaults to None.
        """
        self.X = X
        self.y = y
        self.df_train = df_train
        self.save_path = save_path
        self.model = model
        self.rgo_age = False
        self.whole_X_train_preprocessor = None
        self.whole_age_poly = None
        self.whole_age_lr = None
        if shared_data_dir and not os.path.exists(shared_data_dir):
             try:
                 os.makedirs(shared_data_dir)
                 print(f"Created shared data directory: {shared_data_dir}")
             except OSError as e:
                 print(f"Warning: Could not create shared data directory {shared_data_dir}: {e}")
        self.shared_data_dir = shared_data_dir

    def generate_kfold_finalized(self, y=None, n_splits=5, random_state=0, stratified=False, n_repeats=1):
        """Generates K-Fold or Stratified K-Fold indices."""
        output = []
        X_data = self.df_train
        y_data = self.df_train[y] if y is not None else None

        if stratified and (y is not None):
            if n_repeats > 1: kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else: kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for idx, (train_index, test_index) in enumerate(kf.split(X_data, y_data)):
                output.append(((idx // n_splits, idx % n_splits), train_index, test_index))
        else:
            if n_repeats > 1: kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else: kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for idx, (train_index, test_index) in enumerate(kf.split(X_data)):
                output.append(((idx // n_splits, idx % n_splits), train_index, test_index))
        return output

    def xgboost_objective(self, trial, X_train, y_train):
        """
        Optuna objective function. Called sequentially.
        Launches NESTED parallel jobs for cross_val_score.
        """
        param = {
            'tree_method': 'hist', 'lambda': trial.suggest_loguniform('lambda', 1e-3, 100.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 100.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': 1 # XGBoost runs sequentially
        }
        model = xgb.XGBRegressor(**param, random_state=42, enable_categorical=True)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # cross_val_score runs with n_jobs=-1, using the nested backend
        print(f"    [Optuna Trial {trial.number}] Launching nested parallel cross_val_score...")
        start_cv_time = time.time()
        scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=cv, n_jobs=-1)
        print(f"    [Optuna Trial {trial.number}] Nested cross_val_score finished in {time.time() - start_cv_time:.2f} sec. Result: {np.mean(scores):.4f}")
        return np.mean(scores)

    def xgboost_cross_val_iteration(self, repeat_fold, train_index, test_index):
        """
        Performs a single cross-validation iteration. Runs SEQUENTIALLY.
        Optuna trials run sequentially within this step.
        cross_val_score within Optuna objective runs in PARALLEL (nested).
        """
        # NOTE: This function itself runs sequentially in xgboost_pipe
        print(f"\nStarting Sequential Step - Repeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1}")
        start_time = time.time()
        X_train_df = self.df_train.iloc[train_index][self.X]
        y_train = self.df_train.iloc[train_index][self.y]
        X_test_df = self.df_train.iloc[test_index][self.X]
        y_test = self.df_train.iloc[test_index][self.y]
        numeric_features = [f for f in self.X if self.df_train[f].dtype.name in ['int64', 'float64']]
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')
        preprocessor.set_output(transform='pandas')

        try:
            X_train = preprocessor.fit_transform(X_train_df); X_train.columns = [c.split('__')[-1] for c in X_train.columns]
            X_test = preprocessor.transform(X_test_df); X_test.columns = [c.split('__')[-1] for c in X_test.columns]
            print(f"[{repeat_fold}] Preprocessing successful.")
        except Exception as e:
            print(f"Error during preprocessing in fold {repeat_fold}: {e}")
            return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': {}, 'test_r2': -np.inf, 'error': str(e)}

        if self.rgo_age:
            print(f"[{repeat_fold}] Regressing out age effect...")
            age_train = self.df_train.iloc[train_index]['Age_at_Scan'].values.reshape(-1, 1)
            age_test = self.df_train.iloc[test_index]['Age_at_Scan'].values.reshape(-1, 1)
            poly = PolynomialFeatures(degree=2); age_train_poly = poly.fit_transform(age_train); age_test_poly = poly.transform(age_test)
            lr = LinearRegression(); lr.fit(age_train_poly, y_train)
            print(f"[{repeat_fold}]   Age Regressor R2 (Train): {lr.score(age_train_poly, y_train):.4f}")
            print(f"[{repeat_fold}]   Age Regressor R2 (Test): {lr.score(age_test_poly, y_test):.4f}")
            y_train = y_train - lr.predict(age_train_poly); y_test = y_test - lr.predict(age_test_poly)

        print(f"[{repeat_fold}] Starting sequential hyperparameter optimization (Optuna trials)...")
        optuna_start = time.time()
        sampler = optuna.samplers.TPESampler(seed=42)
        study_name = f"study_repeat{repeat_fold[0]}_fold{repeat_fold[1]}"
        study = optuna.create_study(direction='maximize', sampler=sampler, study_name=study_name)

        try:
            # Optuna runs trials sequentially (n_jobs=1)
            # The objective function triggers nested parallel cross_val_score
            study.optimize(lambda trial: self.xgboost_objective(trial, X_train, y_train), n_trials=150, n_jobs=1)
            best_params = study.best_params; best_value = study.best_value
            print(f"[{repeat_fold}] Optuna finished (sequential trials) in {time.time() - optuna_start:.2f} sec. Best Obj: {best_value:.4f}")
            print(f"[{repeat_fold}] Best Params: {best_params}")
        except Exception as e:
             print(f"Error during Optuna optimization in fold {repeat_fold}: {e}")
             return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': {}, 'test_r2': -np.inf, 'error': f"Optuna failed: {e}"}

        print(f"[{repeat_fold}] Training final model for the fold...")
        fit_start = time.time()
        final_model = xgb.XGBRegressor(**best_params, random_state=42, enable_categorical=True, n_jobs=1) # n_jobs=1
        final_model.fit(X_train, y_train)
        print(f"[{repeat_fold}] Fold model fitting finished in {time.time() - fit_start:.2f} sec.")

        preds = final_model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        try: corr = np.corrcoef(y_test, preds)[0, 1] if len(np.unique(y_test)) > 1 and len(np.unique(preds)) > 1 else np.nan
        except ValueError: corr = np.nan
        try: corr_spearman, _ = spearmanr(y_test, preds) if len(np.unique(y_test)) > 1 and len(np.unique(preds)) > 1 else (np.nan, np.nan)
        except ValueError: corr_spearman = np.nan
        print(f"[{repeat_fold}] Metrics: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Corr={corr:.4f}, Spearman={corr_spearman:.4f}")

        fold_time = time.time() - start_time
        print(f"Finished Sequential Step - Repeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1} in {fold_time:.2f} seconds\n")
        return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': best_params, 'test_r2': r2,
                'test_neg_root_mean_squared_error': rmse, 'test_neg_mean_absolute_error': mae,
                'test_r_corr': corr, 'test_spearmanr': corr_spearman, 'fold_time_seconds': fold_time}

    def xgboost_pipe(self, stratified_label=None, n_repeats=10, n_splits=5):
        """
        Main pipeline: Outer CV Folds run SEQUENTIALLY.
        cross_val_score within Optuna objective runs as NESTED PARALLEL jobs.
        """
        if not HTCONDOR_AVAILABLE:
             print("ERROR: HTCondor backend not available for nested parallelism.")
             return pd.DataFrame(), None
        elif self.shared_data_dir is None:
            print("ERROR: `shared_data_dir` must be specified.")
            return pd.DataFrame(), None
        else:
            backend_to_use = 'htcondor'
            print(f"Using HTCondor backend (Inner Parallel Only). Shared data: {self.shared_data_dir}")

        print(f"Generating {n_repeats} repeats of {n_splits}-fold CV splits...")
        stratified = stratified_label is not None
        cv_splitter = self.generate_kfold_finalized(y=stratified_label, n_splits=n_splits, random_state=42, stratified=stratified, n_repeats=n_repeats)
        num_steps = len(cv_splitter)
        print(f"Total sequential steps (CV folds) to execute: {num_steps}")

        if isinstance(self.y, str) and 'rgo_age' in self.y:
            print("Age regression enabled...")
            original_y = self.y
            if 'Stroop_Test' in self.y: self.y = 'Stroop_Test'
            elif 'Memory_Test' in self.y: self.y = 'Memory_Test'
            else:
                 base_y = self.y.replace('_rgo_age', ''); self.y = base_y if base_y in self.df_train.columns else original_y
            self.rgo_age = True
            if 'Age_at_Scan' in self.X: self.X = [f for f in self.X if f != 'Age_at_Scan']
            print(f"Target: {self.y}, Features: {self.X}")

        scores_list = []
        total_start_time = time.time()
        print("Starting sequential execution (Outer CV Folds) with nested parallelism...")
        log_dir = os.path.join(self.shared_data_dir, f"joblib_logs_{self.y}_inner")
        if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

        # Configure parallel_config to ENABLE nesting for inner cross_val_score
        # Outer loop runs sequentially, so outer throttle is effectively 1.
        # Inner throttle controls parallel cross_val_score folds.
        with parallel_config(
            backend="htcondor",
            pool="head2.htc.inm7.de", request_cpus=1, request_disk="1GB", request_memory="2Gb",
            # Throttle: 1 outer (sequential), 50 inner (cross_val_score folds)
            throttle=[1, 50],
            export_metadata=True,
            shared_data_dir=self.shared_data_dir, log_dir_prefix=log_dir,
            max_recursion_level=1,   # IMPORTANT: Enable nesting
            # n_jobs/pre_dispatch apply to the context, allowing nested calls
            n_jobs=-1, pre_dispatch='all',
        ) as config:
            print(f"Executing {num_steps} outer steps sequentially, nesting enabled (throttle={config.get('throttle')})...")
            # --- SEQUENTIAL Outer Loop ---
            for repeat_fold, train_index, test_index in cv_splitter:
                # Call the function directly (sequentially)
                # The parallel_config context allows the nested cross_val_score to use HTCondor
                result = self.xgboost_cross_val_iteration(repeat_fold, train_index, test_index)
                scores_list.append(result)
            # -----------------------------

        total_end_time = time.time()
        print(f"\nSequential execution with nested parallelism finished in {total_end_time - total_start_time:.2f} seconds.")

        valid_scores = [s for s in scores_list if s and 'error' not in s]
        error_scores = [s for s in scores_list if not s or 'error' in s]
        if not valid_scores: print("ERROR: All steps failed."); return pd.DataFrame(), None
        if error_scores: print(f"Warning: {len(error_scores)} steps failed.")

        scores_df = pd.DataFrame(valid_scores); scores_df.sort_values(by='test_r2', ascending=False, inplace=True)
        if scores_df.empty: print("ERROR: No valid scores."); return pd.DataFrame(), None

        overall_best_params = scores_df.iloc[0]['params']
        print("\nCV results summary:"); print(scores_df[['repeat', 'fold', 'test_r2']].describe())
        print(f"\nOverall best parameters: {overall_best_params}")

        print("Refitting final model sequentially...")
        refit_start = time.time()
        numeric_features = [f for f in self.X if self.df_train[f].dtype.name in ['int64', 'float64']]
        self.whole_X_train_preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')
        self.whole_X_train_preprocessor.set_output(transform='pandas')
        X_train_full = self.df_train[self.X]; y_train_full = self.df_train[self.y].copy()
        X_train_full_processed = self.whole_X_train_preprocessor.fit_transform(X_train_full)
        X_train_full_processed.columns = [c.split('__')[-1] for c in X_train_full_processed.columns]

        if self.rgo_age:
            print("Applying age regression to full dataset...")
            age_train_full = self.df_train['Age_at_Scan'].values.reshape(-1, 1)
            self.whole_age_poly = PolynomialFeatures(degree=2); age_train_full_poly = self.whole_age_poly.fit_transform(age_train_full)
            self.whole_age_lr = LinearRegression(); self.whole_age_lr.fit(age_train_full_poly, y_train_full)
            print(f"  Full data Age Regressor R2: {self.whole_age_lr.score(age_train_full_poly, y_train_full):.4f}")
            y_train_full_final = y_train_full - self.whole_age_lr.predict(age_train_full_poly)
        else: y_train_full_final = y_train_full

        final_best_model = xgb.XGBRegressor(**overall_best_params, random_state=42, enable_categorical=True, n_jobs=-1) # Use local cores
        final_best_model.fit(X_train_full_processed, y_train_full_final)
        print(f"Final model refitting finished in {time.time() - refit_start:.2f} seconds.")

        self.model = final_best_model
        if self.save_path:
            try:
                os.makedirs(self.save_path, exist_ok=True)
                scores_filename = os.path.join(self.save_path, f"cv_scores_{self.y}_inner_parallel.csv")
                scores_df.to_csv(scores_filename, index=False)
                print(f"CV scores saved to {scores_filename}")
            except Exception as e: print(f"Warning: Could not save results: {e}")
        return scores_df, self.model

# %%
"""
Parallel Both Loops (Outer CV & Inner Optuna)
Pipeline with joblib-htcondor integration supporting nested parallelism.
Configuration:
- Outer Loop (Cross-Validation Folds): Parallel via HTCondor
- Middle Loop (Optuna Trials): Sequential within each Outer Job
- Inner Loop (cross_val_score within Optuna Objective): Parallel via HTCondor (Nested)
"""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
)
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import os

# --- Joblib and HTCondor Imports ---
from joblib import Parallel, delayed, parallel_config, __version__ as joblib_version
try:
    from joblib_htcondor import register_htcondor
    register_htcondor()
    HTCONDOR_AVAILABLE = True
    print(f"joblib-htcondor backend registered successfully (joblib version: {joblib_version}).")
except ImportError:
    HTCONDOR_AVAILABLE = False
    print("Warning: joblib-htcondor not found. Nested parallelism requires this backend.")
    print("Install it using: pip install joblib-htcondor")
# ------------------------------------

class AdMLPipeline:
    def __init__(self, X, y, df_train=None, save_path=None, model=None, shared_data_dir=None):
        """
        Initializes the pipeline.

        Args:
            X (list): List of feature column names.
            y (str): Target column name.
            df_train (pd.DataFrame, optional): Training dataframe. Defaults to None.
            save_path (str, optional): Path to save results/models. Defaults to None.
            model (object, optional): Pre-trained model. Defaults to None.
            shared_data_dir (str, optional): Path to a directory accessible by all HTCondor nodes
                                             for joblib data sharing. Required for HTCondor backend.
                                             Defaults to None.
        """
        self.X = X
        self.y = y
        self.df_train = df_train
        self.save_path = save_path
        self.model = model
        self.rgo_age = False
        self.whole_X_train_preprocessor = None
        self.whole_age_poly = None
        self.whole_age_lr = None
        if shared_data_dir and not os.path.exists(shared_data_dir):
             try:
                 os.makedirs(shared_data_dir)
                 print(f"Created shared data directory: {shared_data_dir}")
             except OSError as e:
                 print(f"Warning: Could not create shared data directory {shared_data_dir}: {e}")
        self.shared_data_dir = shared_data_dir

    def generate_kfold_finalized(self, y=None, n_splits=5, random_state=0, stratified=False, n_repeats=1):
        """Generates K-Fold or Stratified K-Fold indices."""
        output = []
        X_data = self.df_train
        y_data = self.df_train[y] if y is not None else None

        if stratified and (y is not None):
            if n_repeats > 1: kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else: kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for idx, (train_index, test_index) in enumerate(kf.split(X_data, y_data)):
                output.append(((idx // n_splits, idx % n_splits), train_index, test_index))
        else:
            if n_repeats > 1: kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else: kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for idx, (train_index, test_index) in enumerate(kf.split(X_data)):
                output.append(((idx // n_splits, idx % n_splits), train_index, test_index))
        return output

    def xgboost_objective(self, trial, X_train, y_train):
        """
        Optuna objective function. Called sequentially by Optuna within an outer parallel job.
        Launches NESTED parallel jobs for cross_val_score.
        """
        param = {
            'tree_method': 'hist', 'lambda': trial.suggest_loguniform('lambda', 1e-3, 100.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 100.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': 1 # XGBoost runs sequentially
        }
        model = xgb.XGBRegressor(**param, random_state=42, enable_categorical=True)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # cross_val_score runs with n_jobs=-1, using the nested backend
        print(f"    [Optuna Trial {trial.number}] Launching nested parallel cross_val_score...")
        start_cv_time = time.time()
        scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=cv, n_jobs=-1)
        print(f"    [Optuna Trial {trial.number}] Nested cross_val_score finished in {time.time() - start_cv_time:.2f} sec. Result: {np.mean(scores):.4f}")
        return np.mean(scores)

    def xgboost_cross_val_iteration(self, repeat_fold, train_index, test_index):
        """
        Performs a single cross-validation iteration. Runs as a PARALLEL outer job.
        Runs Optuna trials sequentially within this job.
        cross_val_score within Optuna objective runs in PARALLEL (nested).
        """
        print(f"\nStarting Outer Job - Repeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1}")
        start_time = time.time()
        X_train_df = self.df_train.iloc[train_index][self.X]
        y_train = self.df_train.iloc[train_index][self.y]
        X_test_df = self.df_train.iloc[test_index][self.X]
        y_test = self.df_train.iloc[test_index][self.y]
        numeric_features = [f for f in self.X if self.df_train[f].dtype.name in ['int64', 'float64']]
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')
        preprocessor.set_output(transform='pandas')

        try:
            X_train = preprocessor.fit_transform(X_train_df); X_train.columns = [c.split('__')[-1] for c in X_train.columns]
            X_test = preprocessor.transform(X_test_df); X_test.columns = [c.split('__')[-1] for c in X_test.columns]
            print(f"[{repeat_fold}] Preprocessing successful.")
        except Exception as e:
            print(f"Error during preprocessing in fold {repeat_fold}: {e}")
            return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': {}, 'test_r2': -np.inf, 'error': str(e)} # Simplified error return

        if self.rgo_age:
            print(f"[{repeat_fold}] Regressing out age effect...")
            age_train = self.df_train.iloc[train_index]['Age_at_Scan'].values.reshape(-1, 1)
            age_test = self.df_train.iloc[test_index]['Age_at_Scan'].values.reshape(-1, 1)
            poly = PolynomialFeatures(degree=2); age_train_poly = poly.fit_transform(age_train); age_test_poly = poly.transform(age_test)
            lr = LinearRegression(); lr.fit(age_train_poly, y_train)
            print(f"[{repeat_fold}]   Age Regressor R2 (Train): {lr.score(age_train_poly, y_train):.4f}")
            print(f"[{repeat_fold}]   Age Regressor R2 (Test): {lr.score(age_test_poly, y_test):.4f}")
            y_train = y_train - lr.predict(age_train_poly); y_test = y_test - lr.predict(age_test_poly)

        print(f"[{repeat_fold}] Starting sequential hyperparameter optimization (Optuna trials)...")
        optuna_start = time.time()
        sampler = optuna.samplers.TPESampler(seed=42)
        study_name = f"study_repeat{repeat_fold[0]}_fold{repeat_fold[1]}"
        study = optuna.create_study(direction='maximize', sampler=sampler, study_name=study_name)

        try:
            # Optuna runs trials sequentially (n_jobs=1)
            # The objective function triggers nested parallel cross_val_score
            study.optimize(lambda trial: self.xgboost_objective(trial, X_train, y_train), n_trials=150, n_jobs=1)
            best_params = study.best_params; best_value = study.best_value
            print(f"[{repeat_fold}] Optuna finished (sequential trials) in {time.time() - optuna_start:.2f} sec. Best Obj: {best_value:.4f}")
            print(f"[{repeat_fold}] Best Params: {best_params}")
        except Exception as e:
             print(f"Error during Optuna optimization in fold {repeat_fold}: {e}")
             return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': {}, 'test_r2': -np.inf, 'error': f"Optuna failed: {e}"}

        print(f"[{repeat_fold}] Training final model for the fold...")
        fit_start = time.time()
        final_model = xgb.XGBRegressor(**best_params, random_state=42, enable_categorical=True, n_jobs=1) # n_jobs=1
        final_model.fit(X_train, y_train)
        print(f"[{repeat_fold}] Fold model fitting finished in {time.time() - fit_start:.2f} sec.")

        preds = final_model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        try: corr = np.corrcoef(y_test, preds)[0, 1] if len(np.unique(y_test)) > 1 and len(np.unique(preds)) > 1 else np.nan
        except ValueError: corr = np.nan
        try: corr_spearman, _ = spearmanr(y_test, preds) if len(np.unique(y_test)) > 1 and len(np.unique(preds)) > 1 else (np.nan, np.nan)
        except ValueError: corr_spearman = np.nan
        print(f"[{repeat_fold}] Metrics: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Corr={corr:.4f}, Spearman={corr_spearman:.4f}")

        fold_time = time.time() - start_time
        print(f"Finished Outer Job - Repeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1} in {fold_time:.2f} seconds\n")
        return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': best_params, 'test_r2': r2,
                'test_neg_root_mean_squared_error': rmse, 'test_neg_mean_absolute_error': mae,
                'test_r_corr': corr, 'test_spearmanr': corr_spearman, 'fold_time_seconds': fold_time}

    def xgboost_pipe(self, stratified_label=None, n_repeats=10, n_splits=5):
        """
        Main pipeline: Outer CV Folds run in PARALLEL.
        Optuna trials run sequentially within folds.
        cross_val_score within Optuna objective runs as NESTED PARALLEL jobs.
        """
        if not HTCONDOR_AVAILABLE:
             print("ERROR: HTCondor backend not available for nested parallelism.")
             return pd.DataFrame(), None
        elif self.shared_data_dir is None:
            print("ERROR: `shared_data_dir` must be specified.")
            return pd.DataFrame(), None
        else:
            backend_to_use = 'htcondor'
            print(f"Using HTCondor backend (Parallel Outer CV & Inner cross_val_score). Shared data: {self.shared_data_dir}")

        print(f"Generating {n_repeats} repeats of {n_splits}-fold CV splits...")
        stratified = stratified_label is not None
        cv_splitter = self.generate_kfold_finalized(y=stratified_label, n_splits=n_splits, random_state=42, stratified=stratified, n_repeats=n_repeats)
        num_jobs = len(cv_splitter)
        print(f"Total outer jobs (CV folds) to submit: {num_jobs}")

        if isinstance(self.y, str) and 'rgo_age' in self.y:
            print("Age regression enabled...")
            original_y = self.y
            if 'Stroop_Test' in self.y: self.y = 'Stroop_Test'
            elif 'Memory_Test' in self.y: self.y = 'Memory_Test'
            else:
                 base_y = self.y.replace('_rgo_age', ''); self.y = base_y if base_y in self.df_train.columns else original_y
            self.rgo_age = True
            if 'Age_at_Scan' in self.X: self.X = [f for f in self.X if f != 'Age_at_Scan']
            print(f"Target: {self.y}, Features: {self.X}")

        scores_list = []
        total_start_time = time.time()
        print("Starting nested parallel execution (Outer: CV Folds, Inner: cross_val_score)...")
        log_dir = os.path.join(self.shared_data_dir, f"joblib_logs_{self.y}_outer_cv_inner_cvs") # Adjusted log dir name
        if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

        # Configure parallel_config for nested execution
        # max_recursion_level=1 enables nesting for the inner cross_val_score
        # throttle=[25, 50] limits outer jobs to 25, inner (cross_val_score) jobs to 50
        with parallel_config(
            backend="htcondor",
            pool="head2.htc.inm7.de", request_cpus=1, request_disk="1GB", request_memory="2Gb",
            throttle=[25, 50], # Throttle: 25 outer (CV), 50 inner (cross_val_score folds)
            export_metadata=True,
            shared_data_dir=self.shared_data_dir, log_dir_prefix=log_dir,
            max_recursion_level=1,   # IMPORTANT: Enable nesting
            n_jobs=-1, pre_dispatch='all',
        ) as config:
            print(f"Submitting {num_jobs} outer CV jobs with nesting enabled (throttle={config.get('throttle')})...")
            # --- PARALLEL Outer Loop ---
            scores_list = Parallel()(
                delayed(self.xgboost_cross_val_iteration)(repeat_fold, train_index, test_index)
                for repeat_fold, train_index, test_index in cv_splitter
            )
            # --------------------------

        total_end_time = time.time()
        print(f"\nNested parallel execution finished in {total_end_time - total_start_time:.2f} seconds.")

        valid_scores = [s for s in scores_list if s and 'error' not in s]
        error_scores = [s for s in scores_list if not s or 'error' in s]
        if not valid_scores: print("ERROR: All folds failed."); return pd.DataFrame(), None
        if error_scores: print(f"Warning: {len(error_scores)} folds failed.")

        scores_df = pd.DataFrame(valid_scores); scores_df.sort_values(by='test_r2', ascending=False, inplace=True)
        if scores_df.empty: print("ERROR: No valid scores."); return pd.DataFrame(), None

        overall_best_params = scores_df.iloc[0]['params']
        print("\nCV results summary:"); print(scores_df[['repeat', 'fold', 'test_r2']].describe())
        print(f"\nOverall best parameters: {overall_best_params}")

        print("Refitting final model sequentially...")
        refit_start = time.time()
        numeric_features = [f for f in self.X if self.df_train[f].dtype.name in ['int64', 'float64']]
        self.whole_X_train_preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')
        self.whole_X_train_preprocessor.set_output(transform='pandas')
        X_train_full = self.df_train[self.X]; y_train_full = self.df_train[self.y].copy()
        X_train_full_processed = self.whole_X_train_preprocessor.fit_transform(X_train_full)
        X_train_full_processed.columns = [c.split('__')[-1] for c in X_train_full_processed.columns]

        if self.rgo_age:
            print("Applying age regression to full dataset...")
            age_train_full = self.df_train['Age_at_Scan'].values.reshape(-1, 1)
            self.whole_age_poly = PolynomialFeatures(degree=2); age_train_full_poly = self.whole_age_poly.fit_transform(age_train_full)
            self.whole_age_lr = LinearRegression(); self.whole_age_lr.fit(age_train_full_poly, y_train_full)
            print(f"  Full data Age Regressor R2: {self.whole_age_lr.score(age_train_full_poly, y_train_full):.4f}")
            y_train_full_final = y_train_full - self.whole_age_lr.predict(age_train_full_poly)
        else: y_train_full_final = y_train_full

        final_best_model = xgb.XGBRegressor(**overall_best_params, random_state=42, enable_categorical=True, n_jobs=-1) # Use local cores
        final_best_model.fit(X_train_full_processed, y_train_full_final)
        print(f"Final model refitting finished in {time.time() - refit_start:.2f} seconds.")

        self.model = final_best_model
        if self.save_path:
            try:
                os.makedirs(self.save_path, exist_ok=True)
                scores_filename = os.path.join(self.save_path, f"cv_scores_{self.y}_outerCV_innerCVS_parallel.csv") # Adjusted filename
                scores_df.to_csv(scores_filename, index=False)
                print(f"CV scores saved to {scores_filename}")
            except Exception as e: print(f"Warning: Could not save results: {e}")
        return scores_df, self.model

# %%
"""
Conditional Parallelism based on Feature Count
Pipeline with joblib-htcondor integration supporting nested parallelism.
Configuration is CONDITIONAL based on number of features:
- If num_features < 20:
    - Outer Loop (Cross-Validation Folds): Parallel via HTCondor (max_recursion_level=0)
    - Middle Loop (Optuna Trials): Sequential within each Outer Job
    - Inner Loop (cross_val_score): Sequential within each Middle Trial (n_jobs=1)
- If num_features >= 20:
    - Outer Loop (Cross-Validation Folds): Parallel via HTCondor (max_recursion_level=1)
    - Middle Loop (Optuna Trials): Sequential within each Outer Job
    - Inner Loop (cross_val_score): Parallel via HTCondor (Nested, n_jobs=-1)
"""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
)
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import os

# --- Joblib and HTCondor Imports ---
from joblib import Parallel, delayed, parallel_config, __version__ as joblib_version
try:
    from joblib_htcondor import register_htcondor
    register_htcondor()
    HTCONDOR_AVAILABLE = True
    print(f"joblib-htcondor backend registered successfully (joblib version: {joblib_version}).")
except ImportError:
    HTCONDOR_AVAILABLE = False
    print("Warning: joblib-htcondor not found. Parallelism requires this backend.")
    print("Install it using: pip install joblib-htcondor")
# ------------------------------------

class AdMLPipeline:
    def __init__(self, X, y, df_train=None, save_path=None, model=None, shared_data_dir=None):
        """
        Initializes the pipeline.

        Args:
            X (list): List of feature column names.
            y (str): Target column name.
            df_train (pd.DataFrame, optional): Training dataframe. Defaults to None.
            save_path (str, optional): Path to save results/models. Defaults to None.
            model (object, optional): Pre-trained model. Defaults to None.
            shared_data_dir (str, optional): Path to a directory accessible by all HTCondor nodes
                                             for joblib data sharing. Required for HTCondor backend.
                                             Defaults to None.
        """
        self.X = X # Initial feature list
        self.y = y
        self.df_train = df_train
        self.save_path = save_path
        self.model = model
        self.rgo_age = False
        self.whole_X_train_preprocessor = None
        self.whole_age_poly = None
        self.whole_age_lr = None
        # Store the original feature list before potential modification (e.g., removing Age)
        self._original_X = list(X)
        if shared_data_dir and not os.path.exists(shared_data_dir):
             try:
                 os.makedirs(shared_data_dir)
                 print(f"Created shared data directory: {shared_data_dir}")
             except OSError as e:
                 print(f"Warning: Could not create shared data directory {shared_data_dir}: {e}")
        self.shared_data_dir = shared_data_dir

    def generate_kfold_finalized(self, y=None, n_splits=5, random_state=0, stratified=False, n_repeats=1):
        """Generates K-Fold or Stratified K-Fold indices."""
        output = []
        X_data = self.df_train
        y_data = self.df_train[y] if y is not None else None

        if stratified and (y is not None):
            if n_repeats > 1: kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else: kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for idx, (train_index, test_index) in enumerate(kf.split(X_data, y_data)):
                output.append(((idx // n_splits, idx % n_splits), train_index, test_index))
        else:
            if n_repeats > 1: kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else: kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for idx, (train_index, test_index) in enumerate(kf.split(X_data)):
                output.append(((idx // n_splits, idx % n_splits), train_index, test_index))
        return output

    def xgboost_objective(self, trial, X_train, y_train):
        """
        Optuna objective function. Called sequentially by Optuna.
        Launches cross_val_score jobs based on the number of features in self.X.
        """
        # Determine if inner CV should be parallel based on the *current* feature list length
        num_features = len(self.X)
        parallelize_inner_cv = num_features >= 20
        inner_cv_n_jobs = -1 if parallelize_inner_cv else 1

        param = {
            'tree_method': 'hist', 'lambda': trial.suggest_loguniform('lambda', 1e-3, 100.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 100.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': 1 # XGBoost model training itself runs sequentially within the job
        }
        model = xgb.XGBRegressor(**param, random_state=42, enable_categorical=True)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Conditionally run cross_val_score in parallel (nested) or sequentially
        print(f"    [Optuna Trial {trial.number}, Feats={num_features}] Launching cross_val_score (Parallel={parallelize_inner_cv}, n_jobs={inner_cv_n_jobs})...")
        start_cv_time = time.time()
        scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=cv, n_jobs=inner_cv_n_jobs)
        print(f"    [Optuna Trial {trial.number}] cross_val_score finished in {time.time() - start_cv_time:.2f} sec. Result: {np.mean(scores):.4f}")
        return np.mean(scores)

    def xgboost_cross_val_iteration(self, repeat_fold, train_index, test_index):
        """
        Performs a single cross-validation iteration. Runs as a PARALLEL outer job.
        Runs Optuna trials sequentially within this job.
        The objective function determines if inner cross_val_score runs in parallel.
        """
        print(f"\nStarting Outer Job - Repeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1}")
        start_time = time.time()
        # Use self.X which might have been modified (e.g., age removed)
        current_features = self.X
        X_train_df = self.df_train.iloc[train_index][current_features]
        y_train = self.df_train.iloc[train_index][self.y]
        X_test_df = self.df_train.iloc[test_index][current_features]
        y_test = self.df_train.iloc[test_index][self.y]

        numeric_features = [f for f in current_features if self.df_train[f].dtype.name in ['int64', 'float64']]
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')
        preprocessor.set_output(transform='pandas')

        try:
            X_train = preprocessor.fit_transform(X_train_df); X_train.columns = [c.split('__')[-1] for c in X_train.columns]
            X_test = preprocessor.transform(X_test_df); X_test.columns = [c.split('__')[-1] for c in X_test.columns]
            print(f"[{repeat_fold}] Preprocessing successful.")
        except Exception as e:
            print(f"Error during preprocessing in fold {repeat_fold}: {e}")
            return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': {}, 'test_r2': -np.inf, 'error': str(e)}

        if self.rgo_age:
            print(f"[{repeat_fold}] Regressing out age effect...")
            # Ensure 'Age_at_Scan' exists before trying to access it
            if 'Age_at_Scan' in self.df_train.columns:
                age_train = self.df_train.iloc[train_index]['Age_at_Scan'].values.reshape(-1, 1)
                age_test = self.df_train.iloc[test_index]['Age_at_Scan'].values.reshape(-1, 1)
                poly = PolynomialFeatures(degree=2); age_train_poly = poly.fit_transform(age_train); age_test_poly = poly.transform(age_test)
                lr = LinearRegression(); lr.fit(age_train_poly, y_train)
                print(f"[{repeat_fold}]   Age Regressor R2 (Train): {lr.score(age_train_poly, y_train):.4f}")
                print(f"[{repeat_fold}]   Age Regressor R2 (Test): {lr.score(age_test_poly, y_test):.4f}")
                y_train = y_train - lr.predict(age_train_poly); y_test = y_test - lr.predict(age_test_poly)
            else:
                print(f"[{repeat_fold}] Warning: 'Age_at_Scan' column not found in df_train, cannot regress out age.")


        print(f"[{repeat_fold}] Starting sequential hyperparameter optimization (Optuna trials)...")
        optuna_start = time.time()
        sampler = optuna.samplers.TPESampler(seed=42)
        study_name = f"study_repeat{repeat_fold[0]}_fold{repeat_fold[1]}"
        study = optuna.create_study(direction='maximize', sampler=sampler, study_name=study_name)

        try:
            # Optuna runs trials sequentially (n_jobs=1)
            # The objective function decides parallelism of inner cross_val_score
            study.optimize(lambda trial: self.xgboost_objective(trial, X_train, y_train), n_trials=150, n_jobs=1)
            best_params = study.best_params; best_value = study.best_value
            print(f"[{repeat_fold}] Optuna finished (sequential trials) in {time.time() - optuna_start:.2f} sec. Best Obj: {best_value:.4f}")
            print(f"[{repeat_fold}] Best Params: {best_params}")
        except Exception as e:
             print(f"Error during Optuna optimization in fold {repeat_fold}: {e}")
             return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': {}, 'test_r2': -np.inf, 'error': f"Optuna failed: {e}"}

        print(f"[{repeat_fold}] Training final model for the fold...")
        fit_start = time.time()
        final_model = xgb.XGBRegressor(**best_params, random_state=42, enable_categorical=True, n_jobs=1) # Final fold model trains sequentially
        final_model.fit(X_train, y_train)
        print(f"[{repeat_fold}] Fold model fitting finished in {time.time() - fit_start:.2f} sec.")

        preds = final_model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        try: corr = np.corrcoef(y_test, preds)[0, 1] if len(np.unique(y_test)) > 1 and len(np.unique(preds)) > 1 else np.nan
        except ValueError: corr = np.nan
        try: corr_spearman, _ = spearmanr(y_test, preds) if len(np.unique(y_test)) > 1 and len(np.unique(preds)) > 1 else (np.nan, np.nan)
        except ValueError: corr_spearman = np.nan
        print(f"[{repeat_fold}] Metrics: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Corr={corr:.4f}, Spearman={corr_spearman:.4f}")

        fold_time = time.time() - start_time
        print(f"Finished Outer Job - Repeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1} in {fold_time:.2f} seconds\n")
        return {'repeat': repeat_fold[0] + 1, 'fold': repeat_fold[1] + 1, 'params': best_params, 'test_r2': r2,
                'test_neg_root_mean_squared_error': rmse, 'test_neg_mean_absolute_error': mae,
                'test_r_corr': corr, 'test_spearmanr': corr_spearman, 'fold_time_seconds': fold_time}

    def xgboost_pipe(self, stratified_label=None, n_repeats=10, n_splits=5):
        """
        Main pipeline: Conditionally enables nested parallelism based on feature count.
        Outer CV Folds always run in PARALLEL.
        Optuna trials always run sequentially within folds.
        Inner cross_val_score runs PARALLEL (nested) only if num_features >= 20.
        """
        if not HTCONDOR_AVAILABLE:
             print("ERROR: HTCondor backend not available.")
             return pd.DataFrame(), None
        elif self.shared_data_dir is None:
            print("ERROR: `shared_data_dir` must be specified.")
            return pd.DataFrame(), None
        else:
            backend_to_use = 'htcondor'
            print(f"Using HTCondor backend. Shared data: {self.shared_data_dir}")

        # --- Handle Age Regression Feature (before counting features) ---
        if isinstance(self.y, str) and 'rgo_age' in self.y:
            print("Age regression requested. Modifying feature list and target.")
            original_y = self.y
            target_set = False
            if 'Stroop_Test' in self.y: self.y = 'Stroop_Test'; target_set = True
            elif 'Memory_Test' in self.y: self.y = 'Memory_Test'; target_set = True
            else:
                 base_y = self.y.replace('_rgo_age', '')
                 if base_y in self.df_train.columns: self.y = base_y; target_set = True
                 else: print(f"Warning: Could not determine base target variable from '{original_y}'.")

            if target_set: # Only modify features if target was successfully set
                self.rgo_age = True
                if 'Age_at_Scan' in self.X:
                    self.X = [f for f in self.X if f != 'Age_at_Scan'] # Update self.X
                    print(f"Target set to: {self.y}")
                    print(f"Features after removing age: {self.X}")
                else:
                    print("Warning: 'Age_at_Scan' not found in features list (self.X), but age regression was requested.")
            else:
                 self.y = original_y # Revert if target wasn't identified
        # -------------------------------------------------------------

        # --- Determine Parallelism Strategy Based on Feature Count ---
        num_features = len(self.X)
        enable_nested_parallelism = num_features >= 20
        max_recursion = 1 if enable_nested_parallelism else 0
        # Use [25] for outer only, [25, 50] for nested
        throttle_config = [25, 50] if enable_nested_parallelism else [25]
        log_suffix = "nested" if enable_nested_parallelism else "outer_only"

        print(f"\nNumber of features = {num_features}.")
        if enable_nested_parallelism:
            print("Strategy: Parallel Outer CV & Parallel Inner cross_val_score (Nested).")
        else:
            print("Strategy: Parallel Outer CV only.")
        print(f"Setting max_recursion_level={max_recursion}, throttle={throttle_config}")
        # -----------------------------------------------------------

        print(f"\nGenerating {n_repeats} repeats of {n_splits}-fold CV splits...")
        stratified = stratified_label is not None
        cv_splitter = self.generate_kfold_finalized(y=stratified_label, n_splits=n_splits, random_state=42, stratified=stratified, n_repeats=n_repeats)
        num_jobs = len(cv_splitter)
        print(f"Total outer jobs (CV folds) to submit: {num_jobs}")

        scores_list = []
        total_start_time = time.time()
        print("Starting parallel execution...")
        log_dir = os.path.join(self.shared_data_dir, f"joblib_logs_{self.y}_{log_suffix}")
        if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

        # Configure parallel_config based on the determined strategy
        with parallel_config(
            backend="htcondor",
            pool="head2.htc.inm7.de", request_cpus=1, request_disk="1GB", request_memory="2Gb",
            throttle=throttle_config, # Use calculated throttle
            export_metadata=True,
            shared_data_dir=self.shared_data_dir, log_dir_prefix=log_dir,
            max_recursion_level=max_recursion,   # Use calculated recursion level
            n_jobs=-1, pre_dispatch='all',
        ) as config:
            print(f"Submitting {num_jobs} outer CV jobs (Nesting Enabled={enable_nested_parallelism}, throttle={config.get('throttle')})...")
            # --- PARALLEL Outer Loop ---
            scores_list = Parallel()(
                delayed(self.xgboost_cross_val_iteration)(repeat_fold, train_index, test_index)
                for repeat_fold, train_index, test_index in cv_splitter
            )
            # --------------------------

        total_end_time = time.time()
        print(f"\nParallel execution finished in {total_end_time - total_start_time:.2f} seconds.")

        # --- Process Results ---
        valid_scores = [s for s in scores_list if s and 'error' not in s]
        error_scores = [s for s in scores_list if not s or 'error' in s]
        if not valid_scores: print("ERROR: All folds failed."); return pd.DataFrame(), None
        if error_scores: print(f"Warning: {len(error_scores)} folds failed.")

        scores_df = pd.DataFrame(valid_scores); scores_df.sort_values(by='test_r2', ascending=False, inplace=True)
        if scores_df.empty: print("ERROR: No valid scores."); return pd.DataFrame(), None

        overall_best_params = scores_df.iloc[0]['params']
        print("\nCV results summary:"); print(scores_df[['repeat', 'fold', 'test_r2']].describe())
        print(f"\nOverall best parameters: {overall_best_params}")
        # ---------------------

        # --- Refit Final Model (Sequentially) ---
        print("Refitting final model sequentially...")
        refit_start = time.time()
        # Use the final self.X for refitting (potentially modified by age regression step)
        numeric_features = [f for f in self.X if self.df_train[f].dtype.name in ['int64', 'float64']]
        self.whole_X_train_preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')
        self.whole_X_train_preprocessor.set_output(transform='pandas')
        X_train_full = self.df_train[self.X]; y_train_full = self.df_train[self.y].copy()
        X_train_full_processed = self.whole_X_train_preprocessor.fit_transform(X_train_full)
        X_train_full_processed.columns = [c.split('__')[-1] for c in X_train_full_processed.columns]

        if self.rgo_age:
            print("Applying age regression to full dataset...")
            # Ensure 'Age_at_Scan' exists before trying to access it
            if 'Age_at_Scan' in self.df_train.columns:
                age_train_full = self.df_train['Age_at_Scan'].values.reshape(-1, 1)
                self.whole_age_poly = PolynomialFeatures(degree=2); age_train_full_poly = self.whole_age_poly.fit_transform(age_train_full)
                self.whole_age_lr = LinearRegression(); self.whole_age_lr.fit(age_train_full_poly, y_train_full)
                print(f"  Full data Age Regressor R2: {self.whole_age_lr.score(age_train_full_poly, y_train_full):.4f}")
                y_train_full_final = y_train_full - self.whole_age_lr.predict(age_train_full_poly)
            else:
                 print("Warning: 'Age_at_Scan' column not found in df_train for final age regression.")
                 y_train_full_final = y_train_full # Proceed without age regression
        else:
             y_train_full_final = y_train_full

        final_best_model = xgb.XGBRegressor(**overall_best_params, random_state=42, enable_categorical=True, n_jobs=-1) # Use local cores for final fit
        final_best_model.fit(X_train_full_processed, y_train_full_final)
        print(f"Final model refitting finished in {time.time() - refit_start:.2f} seconds.")
        # ---------------------------------------

        self.model = final_best_model
        if self.save_path:
            try:
                os.makedirs(self.save_path, exist_ok=True)
                scores_filename = os.path.join(self.save_path, f"cv_scores_{self.y}_conditional_parallel.csv") # Adjusted filename
                scores_df.to_csv(scores_filename, index=False)
                print(f"CV scores saved to {scores_filename}")
            except Exception as e: print(f"Warning: Could not save results: {e}")

        # Restore original feature list if it was modified for age regression
        self.X = self._original_X

        return scores_df, self.model
