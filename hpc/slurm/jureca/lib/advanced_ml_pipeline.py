import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, \
    cross_val_score

from scipy.stats import spearmanr

from autogluon.tabular import TabularPredictor
import xgboost as xgb

import optuna
import shap
from joblib import Parallel, delayed
from datetime import datetime


class AdMLPipeline:
    def __init__(self, X, y, df_train=None, num_cores=None, save_path=None, model=None):
        self.X = X
        self.y = y
        self.df_train = df_train
        self.num_cores = num_cores
        self.save_path = save_path
        self.model = model

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
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_jobs': self.num_cores
        }

        model = xgb.XGBRegressor(**param, random_state=42)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)  # n_splits=5
        scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(r2_score), cv=cv, n_jobs=self.num_cores)
        return np.mean(scores)

    def xgboost_cross_val_iteration(self, repeat_fold, train_index, test_index):
        X_train = self.df_train.iloc[train_index][self.X].values
        y_train = self.df_train.iloc[train_index][self.y].values
        X_test = self.df_train.iloc[test_index][self.X].values
        y_test = self.df_train.iloc[test_index][self.y].values

        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.xgboost_objective(trial, X_train, y_train), n_trials=50)  # n_trials=50
        best_params = study.best_params

        final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=self.num_cores)
        final_model.fit(X_train, y_train)

        preds = final_model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        corr = np.corrcoef(y_test, preds)[0, 1]
        corr_spearman, _ = spearmanr(y_test, preds)

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

        scores_list = []
        for repeat_fold, train_index, test_index in cv_splitter:
            scores_list.append(self.xgboost_cross_val_iteration(repeat_fold, train_index, test_index))

        scores_list.sort(key=lambda x: x['test_r2'], reverse=True)
        overall_best_params = scores_list[0]['params']

        final_best_model = xgb.XGBRegressor(**overall_best_params, random_state=42, n_jobs=self.num_cores)
        final_best_model.fit(self.df_train[self.X], self.df_train[self.y])

        self.model = final_best_model
        scores_df = pd.DataFrame(scores_list)

        return scores_df, self.model

    def autogluon_cross_val_iteration(self, repeat_fold, train_index, test_index):
        repeat_num, fold_num = repeat_fold
        cv_str = f"R{repeat_num + 1}F{fold_num + 1}"

        train_data = self.df_train.iloc[train_index]
        test_data = self.df_train.iloc[test_index]

        # Model training, AutoGluon
        print('\nStart training for fold ' + cv_str + '...')
        predictor_save_path = self.save_path + 'cv_models/' + cv_str + '/'
        if os.path.exists(predictor_save_path):
            shutil.rmtree(predictor_save_path)

        predictor = TabularPredictor(
            label=self.y,
            problem_type='regression',
            eval_metric='r2',
            path=predictor_save_path
        ).fit(
            train_data=train_data[self.X + [self.y]],
            presets='best_quality',
            auto_stack=True,
            ds_args={
                'memory_safe_fits': False,
            },
            refit_full='best',
            keep_only_best=True,  # Keep only the best model (and its ancestors)
            num_cpus=self.num_cores,
            save_space=True,
            # time_limit=60,  # only for testing
        )

        y_true = test_data[self.y]
        y_pred = predictor.predict(test_data[self.X + [self.y]])
        performances = predictor.evaluate(test_data[self.X + [self.y]])
        best_model = predictor.get_model_best()
        leaderboard = predictor.leaderboard(test_data[self.X + [self.y]], silent=True)

        best_model_row_number = leaderboard.index[leaderboard['model'] == best_model].tolist()[0]
        best_score_val = leaderboard.iloc[best_model_row_number]['score_val']
        best_score_test = leaderboard.iloc[best_model_row_number]['score_test']
        models_list = leaderboard['model'].tolist()

        # Calculate metrics
        r2 = performances['r2']
        rmse = performances['root_mean_squared_error']
        mae = performances['mean_absolute_error']
        corr = performances['pearsonr']
        corr_spearman, _ = spearmanr(y_true, y_pred)

        print('Finished training for fold ' + cv_str + '...\n')

        fold_performance = {
            'repeat': repeat_num + 1,
            'fold': fold_num + 1,
            'Best_Model': best_model,
            'score_val': best_score_val,
            'score_test': best_score_test,
            'test_r2': r2,
            'test_neg_root_mean_squared_error': rmse,
            'test_neg_mean_absolute_error': mae,
            'test_r_corr': corr,
            'test_spearmanr': corr_spearman,
            'models_list': str(models_list)
        }

        return pd.DataFrame(fold_performance)

    def run_autogluon_cross_val_iteration(self, stratified_label=None):
        if stratified_label is not None:
            cv_splitter = self.generate_kfold_finalized(y=stratified_label, n_splits=5, random_state=42,
                                                        stratified=True, n_repeats=10)
        else:
            cv_splitter = self.generate_kfold_finalized(y=None, n_splits=5, random_state=42, stratified=False,
                                                        n_repeats=10)

        # execute the autogluon_cross_val_iteration one by one
        performance_dfs = None
        for repeat_fold, train_index, test_index in cv_splitter:
            performance = self.autogluon_cross_val_iteration(repeat_fold, train_index, test_index)
            if performance_dfs is None:
                performance_dfs = performance
            else:
                performance_dfs = pd.concat([performance_dfs, performance])

        # peformance_df = pd.DataFrame(performance_dicts)

        return performance_dfs

    def run_fake_autogluon_cross_val_iteration(self, stratified_label=None):
        """
        If useful and really used, this function should not be uploaded to the github.
        """
        if stratified_label is not None:
            cv_splitter = self.generate_kfold_finalized(y=stratified_label, n_splits=5, random_state=42,
                                                        stratified=True, n_repeats=10)
        else:
            cv_splitter = self.generate_kfold_finalized(y=None, n_splits=5, random_state=42, stratified=False,
                                                        n_repeats=10)

        # performance_dfs = None
        performance_dfs = []
        for repeat_fold, train_index, test_index in cv_splitter:
            repeat_num, fold_num = repeat_fold
            # cv_str = f"R{repeat_num + 1}F{fold_num + 1}"

            train_data = self.df_train.iloc[train_index]
            test_data = self.df_train.iloc[test_index]

            y_true = test_data[self.y]
            y_pred = self.model.predict(test_data[self.X + [self.y]])
            performances = self.model.evaluate(test_data[self.X + [self.y]])

            r2 = performances['r2']
            rmse = performances['root_mean_squared_error']
            mae = performances['mean_absolute_error']
            corr = performances['pearsonr']
            corr_spearman, _ = spearmanr(y_true, y_pred)

            best_model = self.model.get_model_best()
            leaderboard = self.model.leaderboard(test_data[self.X + [self.y]], silent=True)

            best_model_row_number = leaderboard.index[leaderboard['model'] == best_model].tolist()[0]
            best_score_val = leaderboard.iloc[best_model_row_number]['score_val']
            best_score_test = leaderboard.iloc[best_model_row_number]['score_test']
            models_list = leaderboard['model'].tolist()

            fold_performance = {
                'repeat': repeat_num + 1,
                'fold': fold_num + 1,
                'score_val': best_score_val,
                'score_test': best_score_test,
                'test_r2': r2,
                'test_neg_root_mean_squared_error': rmse,
                'test_neg_mean_absolute_error': mae,
                'test_r_corr': corr,
                'test_spearmanr': corr_spearman,
                'Best_Model': best_model,
                'models_list': str(models_list)
            }

            # performance = pd.DataFrame(fold_performance)
            #
            # if performance_dfs is None:
            #     performance_dfs = performance
            # else:
            #     performance_dfs = pd.concat([performance_dfs, performance])
            # Wrap fold_performance in a list to allow creating a DataFrame
            performance = pd.DataFrame([fold_performance])

            performance_dfs.append(performance)

            # Concatenate all DataFrames
        performance_dfs = pd.concat(performance_dfs, ignore_index=True)

        return performance_dfs

    def autogluon_pipe(self):
        predictor_save_path = self.save_path + "autogluon/"  # + datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        # if predictor_save_path already exists, delete the whole folder
        if os.path.exists(predictor_save_path):
            shutil.rmtree(predictor_save_path)

        self.model = TabularPredictor(
            label=self.y,
            problem_type='regression',
            eval_metric='r2',
            path=predictor_save_path
        ).fit(
            train_data=self.df_train[self.X + [self.y]],
            presets='best_quality',
            auto_stack=True,
            # Automatically sets `num_bag_folds` and `num_stack_levels` arguments based on dataset properties.
            ds_args={
                # 'validation_procedure': 'cv',  # Setting to cross-validation
                # 'n_folds': 5,  # Setting the number of folds to 5
                # 'n_repeats': 5,  # Setting the number of repeats to 10?
                'memory_safe_fits': False,
            },
            refit_full='best',
            # set_best_to_refit_full=True,  # Set the refit model as the default for predictions
            keep_only_best=True,  # Keep only the best model (and its ancestors)
            num_cpus=self.num_cores,
            save_space=True,
            # time_limit=60,  # only for testing
            # excluded_model_types=['KNN', 'NN_TORCH'],  # only for testing
        )

        return self.model

    def test_perform_plot(self, df_test, y_test, fig_title=None):
        y_true = y_test
        if isinstance(self.model, TabularPredictor):
            y_pred = self.model.predict(df_test[self.X + [self.y]])
        else:
            y_pred = self.model.predict(df_test[self.X])

        # if model is AutoGluon, execute the following
        if isinstance(self.model, TabularPredictor):
            performances = self.model.evaluate(df_test[self.X + [self.y]])

            # Calculate metrics
            r2 = performances['r2']
            rmse = -performances['root_mean_squared_error']
            mae = -performances['mean_absolute_error']
            corr = performances['pearsonr']
            corr_spearman, _ = spearmanr(y_true, y_pred)

            # get validation score
            best_model = self.model.get_model_best()
            leaderboard = self.model.leaderboard(df_test[self.X + [self.y]], silent=True)
            # print the leaderboard
            print(leaderboard)
            best_model_row_number = leaderboard.index[leaderboard['model'] == best_model].tolist()[0]
            score_val = leaderboard.iloc[best_model_row_number]['score_val']
        else:
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

        if isinstance(self.model, TabularPredictor):
            return score_val, mae, rmse, r2, corr, corr_spearman
        else:
            return mae, rmse, r2, corr, corr_spearman

    def shap_explain(self, explanation, save_path, train_test_case, title=None):
        save_path = save_path + train_test_case + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Summary plot
        shap.plots.beeswarm(explanation, show=False)
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        plt.savefig(save_path + "SHAP_summary.png")
        plt.show()
        plt.close()

        # Bar plot
        shap.plots.bar(explanation, show=False)
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
