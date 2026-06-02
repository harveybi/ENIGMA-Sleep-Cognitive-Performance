import os
import shutil
import copy
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

from autogluon.tabular import TabularPredictor

import shap
import joblib
from joblib import Parallel, delayed
import time
from datetime import datetime

sns.set_context("paper")  # Adjust global font size
plt.rcParams['font.family'] = 'Arial'

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'


class AdMLPipeline:
    def __init__(self, X, y, df_train=None, num_cores=None, num_gpus=0, save_path=None, model=None):
        self.X = X
        self.y = y
        self.df_train = df_train
        self.num_cores = num_cores
        self.num_gpus = num_gpus
        self.save_path = save_path
        self.model = model
        self.rgo_age = False
        # self.whole_X_train_preprocessor = None
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

    def autogluon_cross_val_iteration(self, repeat_fold, train_index, test_index):
        repeat_num, fold_num = repeat_fold
        cv_str = f"R{repeat_num + 1}F{fold_num + 1}"
        print(f"\nStart training for fold {cv_str}...")
        y_train = self.df_train.iloc[train_index][self.y]
        y_test = self.df_train.iloc[test_index][self.y]

        train_data = self.df_train.iloc[train_index]
        test_data = self.df_train.iloc[test_index]

        if self.rgo_age:
            age_train = self.df_train.iloc[train_index]['Age_at_Scan'].values.reshape(-1, 1)
            age_test = self.df_train.iloc[test_index]['Age_at_Scan'].values.reshape(-1, 1)

            # y_train = train_data[self.y]
            # y_test = test_data[self.y]

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

            train_data.loc[:, self.y] = y_train
            test_data.loc[:, self.y] = y_test
            # train_data[self.y] = y_train
            # test_data[self.y] = y_test

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
            # refit_full='best',
            keep_only_best=True,  # Keep only the best model (and its ancestors)
            num_cpus=self.num_cores,
            num_gpus=self.num_gpus,
            save_space=True,
            time_limit=1200,  # small feature size, 60s is enough; not sure about the large feature size
            excluded_model_types=['KNN'],
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

        print(f"y_train mean: {train_data[self.y].mean()}")
        print(f"y_test mean: {y_true.mean()}")
        print(f"pred mean: {y_pred.mean()}")

        print(f"\nRepeat: {repeat_fold[0] + 1}, Fold: {repeat_fold[1] + 1} end\n")

        return pd.DataFrame([fold_performance])

    def run_autogluon_cross_val_iteration(self, stratified_label=None, htcondorcluster=False):
        if stratified_label is not None:
            cv_splitter = self.generate_kfold_finalized(y=stratified_label, n_splits=5, random_state=42,
                                                        stratified=True, n_repeats=10)  # n_repeats=10
        else:
            cv_splitter = self.generate_kfold_finalized(y=None, n_splits=5, random_state=42, stratified=False,
                                                        n_repeats=10)  # n_repeats=10

        if 'Age_at_Scan' in self.X and 'rgo_age' in self.y:
            if 'Stroop_Test' in self.y:
                self.y = 'Stroop_Test'
            elif 'Memory_Test' in self.y:
                self.y = 'Memory_Test'

            self.rgo_age = True

            self.X = copy.deepcopy(self.X)
            self.X.remove('Age_at_Scan')

        # execute the autogluon_cross_val_iteration one by one
        performance_dfs = None
        if htcondorcluster:
            # ------------------- HTCondor parallelization -------------------
            print("\nStart Dask cluster...\n")
            cluster = HTCondorCluster(
                cores=1,
                memory="4GB",  # 8GB
                disk="4GB",
                submit_command_extra=["-name", "head2.htc.inm7.de"],
            )

            cluster.scale(jobs=self.num_cores)
            client = Client(cluster)
            client.wait_for_workers(n_workers=self.num_cores, timeout=2 ** 30)
            start_dask = time.time()

            with joblib.parallel_backend(backend="dask"):
                for repeat_fold, train_index, test_index in cv_splitter:
                    performance = self.autogluon_cross_val_iteration(repeat_fold, train_index, test_index)
                    if performance_dfs is None:
                        performance_dfs = performance
                    else:
                        performance_dfs = pd.concat([performance_dfs, performance])

            end_dask = time.time()
            print(f"\nTime taken for Dask: {end_dask - start_dask:.2f} seconds\n")
            cluster.close()
            client.close()

        else:
            for repeat_fold, train_index, test_index in cv_splitter:
                performance = self.autogluon_cross_val_iteration(repeat_fold, train_index, test_index)
                if performance_dfs is None:
                    performance_dfs = performance
                else:
                    performance_dfs = pd.concat([performance_dfs, performance])

        # peformance_df = pd.DataFrame(performance_dicts)

        return performance_dfs

    def autogluon_pipe(self):
        predictor_save_path = self.save_path + "autogluon/"  # + datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        # if predictor_save_path already exists, delete the whole folder
        if os.path.exists(predictor_save_path):
            shutil.rmtree(predictor_save_path)

        if 'Age_at_Scan' in self.X and 'rgo_age' in self.y:
            if 'Stroop_Test' in self.y:
                self.y = 'Stroop_Test'
            elif 'Memory_Test' in self.y:
                self.y = 'Memory_Test'

            self.rgo_age = True
            # self.X.remove('Age_at_Scan')
            self.X = copy.deepcopy(self.X)
            self.X.remove('Age_at_Scan')

        if self.rgo_age:
            y_train = self.df_train[self.y]
            age_train = self.df_train['Age_at_Scan'].values.reshape(-1, 1)

            # Create polynomial features
            self.whole_age_poly = PolynomialFeatures(degree=2)
            age_train_poly = self.whole_age_poly.fit_transform(age_train)

            self.whole_age_lr = LinearRegression()
            self.whole_age_lr.fit(age_train_poly, y_train)

            y_train = y_train - self.whole_age_lr.predict(age_train_poly)

            self.df_train[self.y] = y_train

            # save the preprocessor
            joblib.dump(self.whole_age_poly, self.save_path + 'whole_age_poly.pkl')
            joblib.dump(self.whole_age_lr, self.save_path + 'whole_age_lr.pkl')

        # new setting for hyperparameters ------------------------------------------------------------------------------
        gpus_for_nn = 1 if getattr(self, "num_gpus", 0) and self.num_gpus > 0 else 0

        # hyperparameters = {
        #     # LightGBM family (both standard and XT) on CPU
        #     "GBM": [
        #         {"extra_trees": True, "ag_args_fit": {"num_gpus": 0}},  # LightGBMXT (CPU)
        #         {"ag_args_fit": {"num_gpus": 0}},  # LightGBM (CPU)
        #     ],
        #     # CatBoost on CPU (GPU hurt quality in your log)
        #     "CAT": {"ag_args_fit": {"num_gpus": 0}},
        #     # Neural nets: allow up to 1 GPU (Torch/FastAI each uses a single GPU)
        #     "NN_TORCH": {"ag_args_fit": {"num_gpus": gpus_for_nn}},
        #     "FASTAI": {"ag_args_fit": {"num_gpus": gpus_for_nn}},
        # }
        """
        Stroop, Sleep_Cov_Brain: GBM, XGB, NN_TORCH, FASTAI
        Memory, Sleep_Cov_Brain: CAT, XGB, NN_TORCH, FASTAI
        
        Stroop, Sleep_Cov: GBM, XGB, NN_TORCH, FASTAI
        Memory, Sleep_Cov:?
        """
        hyperparameters = {

            # Keep both LightGBM variants (XT + standard)
            "GBM": [
                {"extra_trees": True, "ag_args_fit": {"num_gpus": 0}},  # LightGBMXT (CPU)
                {"ag_args_fit": {"num_gpus": 0}},  # LightGBM (CPU)
            ],

            # Keep CatBoost, XGBoost, RF, XT, LR as in preset (CPU by default)
            # "CAT": {"ag_args_fit": {"num_gpus": 0}},
            "XGB": {"ag_args_fit": {"num_gpus": 0}},  # Will use CPU because AG_ARGS_FIT num_gpus=0
            # "RF": [{}],
            # "XT": [{}],

            # Allow neural nets to use up to 1 GPU
            "NN_TORCH": {"ag_args_fit": {"num_gpus": gpus_for_nn}},
            "FASTAI": {"ag_args_fit": {"num_gpus": gpus_for_nn}},
        }

        self.model = TabularPredictor(
            label=self.y,
            problem_type='regression',
            eval_metric='r2',
            path=predictor_save_path
        ).fit(
            train_data=self.df_train[self.X + [self.y]],
            presets='best_quality',
            auto_stack=True,
            num_stack_levels=1,
            dynamic_stacking=False,
            ds_args={
                'memory_safe_fits': False,
            },
            refit_full='best',
            # set_best_to_refit_full=True,  # Set the refit model as the default for predictions
            keep_only_best=True,  # Keep only the best model (and its ancestors)
            save_space=True,
            # If you only care about deploying the most accurate predictor with the smallest file-size and no longer need any of the other trained models or functionality beyond prediction on new data, then set: keep_only_best=True, save_space=True. This is equivalent to calling predictor.delete_models(models_to_keep=’best’, dry_run=False) directly after fit().
            num_cpus=self.num_cores,
            num_gpus=self.num_gpus,
            time_limit=3600,  # 60 only for testing
            excluded_model_types=['KNN'],
            verbosity=2,
            hyperparameters=hyperparameters,
        )

        return self.model

    def test_perform_plot(self, df_test, fig_title=None):
        # X_test = self.whole_X_train_preprocessor.transform(df_test[self.X].values)
        X_test = df_test[self.X].values
        y_test = df_test[self.y].values

        if self.rgo_age:
            age_test = df_test['Age_at_Scan'].values.reshape(-1, 1)

            age_test_poly = self.whole_age_poly.transform(age_test)

            y_test = y_test - self.whole_age_lr.predict(age_test_poly)

            y_true = y_test
            df_test[self.y] = y_test

        else:
            y_true = y_test

        df_test[self.X] = X_test

        y_pred = self.model.predict(df_test[self.X + [self.y]])

        # if model is AutoGluon, execute the following
        performances = self.model.evaluate(df_test[self.X + [self.y]])

        # Calculate metrics
        r2 = performances['r2']
        rmse = -performances['root_mean_squared_error']
        mae = -performances['mean_absolute_error']
        corr = performances['pearsonr']
        corr_spearman, _ = spearmanr(y_true, y_pred)

        # get validation score
        best_model = self.model.model_best
        leaderboard = self.model.leaderboard(df_test[self.X + [self.y]], silent=True)
        # print the leaderboard
        # print(leaderboard)
        best_model_row_number = leaderboard.index[leaderboard['model'] == best_model].tolist()[0]
        score_val = leaderboard.iloc[best_model_row_number]['score_val']

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

        return score_val, mae, rmse, r2, corr, corr_spearman

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
