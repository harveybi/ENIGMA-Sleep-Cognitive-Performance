import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator, TargetPipelineCreator

from sklearn.metrics import make_scorer
from julearn.scoring import register_scorer
from julearn.models import register_model
from scipy.stats import spearmanr

import joblib
from joblib import dump, load, Parallel, delayed, parallel_config
from joblib_htcondor import register_htcondor

import shap
import time
import datetime


def spearman_r_scorer(y_true, y_pred):
    r, _ = spearmanr(y_true, y_pred)
    return r


register_scorer(scorer_name="spearmanr", scorer=make_scorer(spearman_r_scorer))


class MLPipeline:
    def __init__(self, X, y, df_train=None, num_cores=None, save_path=None):
        self.X = X
        self.y = y
        self.df_train = df_train
        # self.df_test = df_test
        self.num_cores = num_cores
        self.save_path = save_path
        # self.y_test = df_test[y].values.reshape(-1) if df_test is not None else None
        self.scores = None
        self.model = None

    def generate_kfold(self, y=None, n_splits=5, random_state=0, stratified=False, n_repeats=1):
        # kf = None
        X_data = self.df_train
        y_data = self.df_train[y] if y is not None else None

        if stratified and (y is not None):
            if n_repeats > 1:
                kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            kf.get_n_splits(X_data, y_data)
            return [[train_index, test_index] for train_index, test_index in kf.split(X_data, y_data)]

        else:
            if n_repeats > 1:
                kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            else:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            kf.get_n_splits(X_data)
            return [[train_index, test_index] for train_index, test_index in kf.split(X_data)]

    def dummy_pipe(self, stratified_label=None):
        X_types = {"features": self.X}
        pipeline_creator = PipelineCreator(problem_type="regression", apply_to="features")
        pipeline_creator.add("zscore")
        pipeline_creator.add("dummy")

        if stratified_label is not None:
            cv_splitter = self.generate_kfold(y=stratified_label, n_splits=5, random_state=42,
                                              stratified=True, n_repeats=10)
        else:
            cv_splitter = self.generate_kfold(y=None, n_splits=5, random_state=42, stratified=False,
                                              n_repeats=10)

        self.scores, self.model = run_cross_validation(
            data=self.df_train,
            X=self.X,
            y=self.y,
            X_types=X_types,
            model=pipeline_creator,
            scoring=["r2", "neg_mean_absolute_error", 'neg_root_mean_squared_error', 'r_corr', 'spearmanr'],
            return_estimator='final',
            return_train_score=True,
            cv=cv_splitter,
            seed=42,
            n_jobs=self.num_cores,
            verbose=2
        )

        return self.scores, self.model

    def linear_rg_pipe(self, stratified_label=None):
        X_types = {"features": self.X}
        pipeline_creator = PipelineCreator(problem_type="regression", apply_to="features")
        pipeline_creator.add("zscore")
        pipeline_creator.add("linreg")

        if stratified_label is not None:
            cv_splitter = self.generate_kfold(y=stratified_label, n_splits=5, random_state=42,
                                              stratified=True, n_repeats=10)
        else:
            cv_splitter = self.generate_kfold(y=None, n_splits=5, random_state=42, stratified=False,
                                              n_repeats=10)

        self.scores, self.model = run_cross_validation(
            data=self.df_train,
            X=self.X,
            y=self.y,
            X_types=X_types,
            model=pipeline_creator,
            scoring=["r2", "neg_mean_absolute_error", 'neg_root_mean_squared_error', 'r_corr', 'spearmanr'],
            return_estimator='final',
            return_train_score=True,
            cv=cv_splitter,
            seed=42,
            n_jobs=self.num_cores,
            verbose=2
        )

        return self.scores, self.model

    def ridge_rg_pipe(self, stratified_label=None):
        X_types = {"features": self.X}
        pipeline_creator = PipelineCreator(problem_type="regression", apply_to="features")
        pipeline_creator.add("zscore")
        pipeline_creator.add("ridge", alphas=np.logspace(-5, 5, num=11))

        if stratified_label is not None:
            cv_splitter = self.generate_kfold(y=stratified_label, n_splits=5, random_state=42,
                                              stratified=True, n_repeats=10)
        else:
            cv_splitter = self.generate_kfold(y=None, n_splits=5, random_state=42, stratified=False,
                                              n_repeats=10)

        self.scores, self.model = run_cross_validation(
            data=self.df_train,
            X=self.X,
            y=self.y,
            X_types=X_types,
            model=pipeline_creator,
            scoring=["r2", "neg_mean_absolute_error", 'neg_root_mean_squared_error', 'r_corr', 'spearmanr'],
            return_estimator='final',
            return_train_score=True,
            cv=cv_splitter,
            seed=42,
            n_jobs=self.num_cores,
            verbose=2
        )

        return self.scores, self.model

    def svm_linear_pipe(self, stratified_label=None):
        X_types = {"features": self.X}
        pipeline_creator = PipelineCreator(problem_type="regression", apply_to="features")
        pipeline_creator.add("zscore")
        pipeline_creator.add(
            "svm",
            kernel=["linear"],
            C=[0.01, 0.1, 1, 10, 100, 1000],
            epsilon=[0.01, 0.1, 0.2, 0.5, 1],
        )

        if stratified_label is not None:
            cv_splitter = self.generate_kfold(y=stratified_label, n_splits=5, random_state=42,
                                              stratified=True, n_repeats=10)
        else:
            cv_splitter = self.generate_kfold(y=None, n_splits=5, random_state=42, stratified=False,
                                              n_repeats=10)

        self.scores, self.model = run_cross_validation(
            data=self.df_train,
            X=self.X,
            y=self.y,
            X_types=X_types,
            model=pipeline_creator,
            scoring=["r2", "neg_mean_absolute_error", 'neg_root_mean_squared_error', 'r_corr', 'spearmanr'],
            return_estimator='final',
            return_train_score=True,
            cv=cv_splitter,
            seed=42,
            n_jobs=self.num_cores,
            verbose=2
        )

        return self.scores, self.model

    def svm_rbf_pipe(self, stratified_label=None):
        X_types = {"features": self.X}
        pipeline_creator = PipelineCreator(problem_type="regression", apply_to="features")
        pipeline_creator.add("zscore")
        pipeline_creator.add(
            "svm",
            kernel=["rbf"],
            C=[0.01, 0.1, 1, 10, 100, 1000],
            gamma=[0.001, 0.01, 0.1, 1, 10, "scale", "auto"],
            epsilon=[0.01, 0.1, 0.2, 0.5, 1],
        )

        # pipeline_creator.add(
        #     "svm",
        #     kernel=["rbf"],
        #     C=[0.01, 0.1],
        # )

        if stratified_label is not None:
            cv_splitter = self.generate_kfold(y=stratified_label, n_splits=5, random_state=42,
                                              stratified=True, n_repeats=10)  # 5, 10
        else:
            cv_splitter = self.generate_kfold(y=None, n_splits=5, random_state=42, stratified=False,
                                              n_repeats=10)  # 5, 10

        self.scores, self.model = run_cross_validation(
            data=self.df_train,
            X=self.X,
            y=self.y,
            X_types=X_types,
            model=pipeline_creator,
            scoring=["r2", "neg_mean_absolute_error", 'neg_root_mean_squared_error', 'r_corr', 'spearmanr'],
            return_estimator='final',
            return_train_score=True,
            cv=cv_splitter,
            seed=42,
            n_jobs=self.num_cores,
            verbose=2
        )

        return self.scores, self.model

    def rf_pipe(self, stratified_label=None):
        X_types = {"features": self.X}
        pipeline_creator = PipelineCreator(problem_type="regression", apply_to="features")
        pipeline_creator.add("zscore")
        pipeline_creator.add(
            "rf",
            n_estimators=[100, 200, 300, 400, 500, 750, 1000],
            max_depth=[5, 10, 20, 30, 40],
            min_samples_split=[2, 5, 10],
            min_samples_leaf=[1, 2, 4],
            max_features=['sqrt', 'log2', None],
            bootstrap=[True, False]
        )

        # pipeline_creator.add(
        #     "rf",
        #     n_estimators=[100, 200],
        #     max_depth=[5, 10],
        # )

        if stratified_label is not None:
            cv_splitter = self.generate_kfold(y=stratified_label, n_splits=5, random_state=42,
                                              stratified=True, n_repeats=10)  # 5, 10
        else:
            cv_splitter = self.generate_kfold(y=None, n_splits=5, random_state=42, stratified=False,
                                              n_repeats=10)  # 5, 10

        # register_htcondor("INFO")  # Set logging level to INFO
        #
        # with parallel_config(
        #         backend="htcondor",
        #         pool="head2.htc.inm7.de",
        #         n_jobs=-1,
        #         request_cpus=1,
        #         request_disk="1GB",
        #         request_memory="4Gb",
        #         shared_data_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/rf/joblib_htcondor/",
        #         max_recursion_level=1,
        #         throttle=[25, 50],
        # ):
        #     self.scores, self.model = run_cross_validation(
        #         data=self.df_train,
        #         X=self.X,
        #         y=self.y,
        #         X_types=X_types,
        #         model=pipeline_creator,
        #         scoring=["r2", "neg_mean_absolute_error", 'neg_root_mean_squared_error', 'r_corr', 'spearmanr'],
        #         return_estimator='final',
        #         return_train_score=True,
        #         cv=cv_splitter,
        #         seed=42,
        #         # n_jobs=self.num_cores,
        #         verbose=2
        #     )

        self.scores, self.model = run_cross_validation(
            data=self.df_train,
            X=self.X,
            y=self.y,
            X_types=X_types,
            model=pipeline_creator,
            scoring=["r2", "neg_mean_absolute_error", 'neg_root_mean_squared_error', 'r_corr', 'spearmanr'],
            return_estimator='final',
            return_train_score=True,
            cv=cv_splitter,
            seed=42,
            # n_jobs=self.num_cores,
            verbose=2
        )

        return self.scores, self.model


    def test_perform_plot(self, df_test, y_test, fig_title=None):
        y_true = y_test
        y_pred = self.model.predict(df_test[self.X])

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        corr = np.corrcoef(y_pred, y_true)[1, 0]
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

        # Summary plot
        shap.plots.beeswarm(explanation, show=False)
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path + "SHAP_summary.png")
        plt.show()
        plt.close()

        # Bar plot
        shap.plots.bar(explanation, show=False)
        if title is not None:
            plt.title(title)
        plt.tight_layout()
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
