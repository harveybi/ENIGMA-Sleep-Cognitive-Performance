import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score

from scipy.stats import spearmanr

import xgboost as xgb

import optuna
import shap
import time

import joblib
from dask.distributed import Client
from dask_jobqueue.htcondor import HTCondorCluster


class AdMLPipeline:
    def __init__(self, X, y, df_train=None, num_cores=None, save_path=None, model=None, htcondorcluster=False):
        self.X = X
        self.y = y
        self.df_train = df_train
        self.num_cores = num_cores
        self.save_path = save_path
        self.model = model
        self.htcondorcluster = htcondorcluster
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
            'n_jobs': self.num_cores
        }

        model = xgb.XGBRegressor(**param, random_state=42, enable_categorical=True)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)  # n_splits=5
        scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=cv, n_jobs=self.num_cores)
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

        # study = optuna.create_study(direction='maximize')
        # study.optimize(lambda trial: self.xgboost_objective(trial, X_train, y_train), n_trials=30)  # n_trials=100
        # best_params = study.best_params
        #
        # # # print description of X_train and y_train's values
        # # print(f"X_train: {X_train.describe()}")
        # # print(f"y_train: {y_train.describe()}")
        #
        # final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=self.num_cores, enable_categorical=True)
        # final_model.fit(X_train, y_train)

        if self.htcondorcluster:
            # ------------------- HTCondor parallelization -------------------
            cluster = HTCondorCluster(
                        cores=1,
                        memory="4GB",  # 8GB
                        disk="2GB",  # 4GB
                        submit_command_extra=["-name", "head2.htc.inm7.de"],
                    )

            cluster.scale(jobs=self.num_cores)
            client = Client(cluster)
            start_dask = time.time()

            with joblib.parallel_backend(backend="dask", wait_for_workers_timeout=2 ** 30):
                sampler = optuna.samplers.TPESampler(seed=42)
                study = optuna.create_study(direction='maximize', sampler=sampler)
                study.optimize(lambda trial: self.xgboost_objective(trial, X_train, y_train), n_trials=150)  # n_trials=150, n_jobs=self.num_cores
                best_params = study.best_params
                final_model = xgb.XGBRegressor(**best_params, random_state=42, enable_categorical=True)  # n_jobs=self.num_cores
                fit_start = time.time()
                final_model.fit(X_train, y_train)
                fit_end = time.time()
                print(f"\nTime taken for fitting: {fit_end - fit_start:.2f} seconds.")

            end_dask = time.time()
            print(f"\nTime taken for Dask: {end_dask - start_dask:.2f} seconds\n")
            cluster.close()
            client.close()
            # ------------------- HTCondor parallelization -------------------
        else:
            # ------------------- Normal parallelization -------------------
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(lambda trial: self.xgboost_objective(trial, X_train, y_train), n_trials=150)

            best_params = study.best_params
            final_model = xgb.XGBRegressor(**best_params, random_state=42, enable_categorical=True)  # n_jobs=self.num_cores
            fit_start = time.time()
            final_model.fit(X_train, y_train)
            fit_end = time.time()
            print(f"\nTime taken for fitting: {fit_end - fit_start:.2f} seconds.")
            # ------------------- Normal parallelization -------------------

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
                                                        stratified=True, n_repeats=10)  # 5, 10
        else:
            cv_splitter = self.generate_kfold_finalized(y=None, n_splits=5, random_state=42, stratified=False,
                                                        n_repeats=10)  # 5, 10

        if 'Age_at_Scan' in self.X and 'rgo_age' in self.y:
            if 'Stroop_Test' in self.y:
                self.y = 'Stroop_Test'
            elif 'Memory_Test' in self.y:
                self.y = 'Memory_Test'

            self.rgo_age = True
            self.X.remove('Age_at_Scan')

        scores_list = []
        for repeat_fold, train_index, test_index in cv_splitter:
            scores_list.append(self.xgboost_cross_val_iteration(repeat_fold, train_index, test_index))

        scores_list.sort(key=lambda x: x['test_r2'], reverse=True)
        overall_best_params = scores_list[0]['params']

        final_best_model = xgb.XGBRegressor(**overall_best_params, random_state=42, enable_categorical=True)  # n_jobs=self.num_cores

        # Define categorical and numeric features
        # categorical_features = [feature for feature in self.X if self.df_train[feature].dtype == 'O']
        numeric_features = [feature for feature in self.X if self.df_train[feature].dtype.name in ['int64', 'float64']]

        # Preprocessor definition
        whole_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                # ('cat', OneHotEncoder(), categorical_features)
            ], remainder='passthrough')
        whole_preprocessor.set_output(transform='pandas')

        self.whole_X_train_preprocessor = whole_preprocessor
        X_train = self.df_train[self.X]
        y_train = self.df_train[self.y]
        X_train = self.whole_X_train_preprocessor.fit_transform(X_train)
        X_train.columns = [column.split('__')[-1] for column in X_train.columns]

        # save the feature preprocessor
        joblib.dump(self.whole_X_train_preprocessor, self.save_path + "whole_X_train_preprocessor.pkl")

        if self.rgo_age:
            age_train = self.df_train['Age_at_Scan'].values.reshape(-1, 1)

            # Create polynomial features
            self.whole_age_poly = PolynomialFeatures(degree=2)
            age_train_poly = self.whole_age_poly.fit_transform(age_train)

            self.whole_age_lr = LinearRegression()
            self.whole_age_lr.fit(age_train_poly, y_train)

            y_train = y_train - self.whole_age_lr.predict(age_train_poly)

            # save the target preprocessor and the linear regression model
            joblib.dump(self.whole_age_poly, self.save_path + "whole_age_poly.pkl")
            joblib.dump(self.whole_age_lr, self.save_path + "whole_age_lr.pkl")

        # final_best_model.fit(self.df_train[self.X], self.df_train[self.y])
        refit_start = time.time()
        final_best_model.fit(X_train, y_train)
        refit_end = time.time()
        print(f"\nTime taken for refit: {refit_end - refit_start:.2f} seconds\n")

        self.model = final_best_model
        scores_df = pd.DataFrame(scores_list)

        return scores_df, self.model

    def test_perform_plot(self, df_test, X_train_preprocessor=None, y_age_poly=None, y_age_lr=None,fig_title=None):
        if X_train_preprocessor is not None:
            self.whole_X_train_preprocessor = X_train_preprocessor
            X_test = self.whole_X_train_preprocessor.transform(df_test[self.X])
            X_test.columns = [column.split('__')[-1] for column in X_test.columns]
        else:
            X_test = self.whole_X_train_preprocessor.transform(df_test[self.X])
            X_test.columns = [column.split('__')[-1] for column in X_test.columns]
            # X_test = df_test[self.X]

        y_test = df_test[self.y].values

        if self.rgo_age:
            if y_age_poly is not None and y_age_lr is not None:
                self.whole_age_poly = y_age_poly
                self.whole_age_lr = y_age_lr
            else:
                pass

            age_test = df_test['Age_at_Scan'].values.reshape(-1, 1)
            age_test_poly = self.whole_age_poly.transform(age_test)
            y_test = y_test - self.whole_age_lr.predict(age_test_poly)

            y_true = y_test

        else:
            y_true = y_test

        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        corr_spearman, _ = spearmanr(y_true, y_pred)

        # Create a DataFrame for Seaborn
        data = pd.DataFrame({
            'True Values': y_true,
            'Predicted Values': y_pred
        })
        # Set plot style
        sns.set_style("darkgrid")
        # Create lmplot
        lm = sns.lmplot(x='True Values', y='Predicted Values', data=data, ci=None, aspect=1.3, height=7)
        # Annotations
        text = f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.2f}  CORR: {corr:.2f}  Spearman: {corr_spearman:.2f}"
        lm.ax.text(0.95, 0.05, text, verticalalignment='bottom', horizontalalignment='right', transform=lm.ax.transAxes,
                   fontsize=12)
        # Adjustments and display
        lm.set_axis_labels("True Values", "Predicted Values")
        # Set plot title
        if fig_title is None:
            fig_title = "Actual vs Predicted"
        else:
            fig_title = fig_title + " - Actual vs Predicted"
        plt.title(fig_title)
        plt.tight_layout()
        if self.save_path is not None:
            plt.savefig(self.save_path + fig_title + ".png")
        plt.show()
        plt.close()

        # create jointplot
        join = sns.jointplot(x='True Values', y='Predicted Values', data=data, kind='reg', height=7)
        join.ax_joint.text(0.95, 0.05, text, verticalalignment='bottom', horizontalalignment='right',
                           transform=join.ax_joint.transAxes, fontsize=12)
        join.set_axis_labels("True Values", "Predicted Values")
        if fig_title is None:
            fig_title = "Actual vs Predicted"
        else:
            fig_title = fig_title + " - Actual vs Predicted"
        plt.title(fig_title)
        plt.tight_layout()
        if self.save_path is not None:
            plt.savefig(self.save_path + fig_title + "_jointplot.png")
        plt.show()
        plt.close()

        return mae, rmse, r2, corr, corr_spearman

    def shap_explain(self, explanation, save_path, train_test_case, title=None):
        save_path = save_path + train_test_case + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # # Summary plot
        # shap.plots.beeswarm(explanation, show=False)
        # plt.tight_layout()
        # if title is not None:
        #     plt.title(title)
        # plt.savefig(save_path + "SHAP_summary.png")
        # plt.show()
        # plt.close()

        # Bar plot
        shap.plots.bar(explanation, clustering_cutoff=0.8, show=False)
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        plt.savefig(save_path + "SHAP_bar.png")
        plt.show()
        plt.close()

        # scatter plot
        importance_df = pd.DataFrame({
            'feature': self.X,
            'importance': np.abs(explanation.values).mean(axis=0)
        })
        # select top 10 features to a list
        top_10_features = importance_df.sort_values(by='importance', ascending=False).head(10)['feature'].tolist()
        for feature in top_10_features:
            for color in ['Age_at_Scan', 'PSG_Sleep_Dur', 'Self_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Eff']:
                # if there's no such feature in X, skip
                if color not in self.X:
                    continue
                shap.plots.scatter(explanation[:, feature], color=explanation[:, color], show=False)
                plt.title(feature)
                plt.tight_layout()
                plt.savefig(save_path + f"{feature}_{color}_scatter.png")
                plt.show()
                plt.close()

        # force plot
        output_of_force_plot = shap.force_plot(explanation, matplotlib=False, show=False)
        file = save_path + train_test_case + '_force_plot_average.html'
        shap.save_html(file, output_of_force_plot)
