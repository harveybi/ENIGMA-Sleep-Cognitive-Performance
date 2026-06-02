# import packages
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats as st

from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge

from julearn import run_cross_validation
from julearn.utils import configure_logging

# %%
class Models:
    """Models for predicting cognitive performance

    This Models will implement the following models:
    1. Linear models: Ridge regression, Lasso regression, ElasticNet regression, SVR linear kernel
    2. Non-linear models: SVR rbf kernel, KRR, Random Forest, Gradient Boosting, LightGBM, XGBoost

    Attributes：

    """

    def __init__(self, df_dataset, X_list, y_list, ml_model, cv_folds=None, repeats=None, confounds_list=None, model_params=None):
        """Initialization of Sleep_Cog_Models.

        Args:

        """
        self.df_dataset = df_dataset
        self.X_list = X_list
        self.y_list = y_list
        self.ml_model = ml_model
        self.model_name = ml_model
        self.repeats = repeats
        self.cv_folds = cv_folds
        self.confounds_list = confounds_list
        self.model_params = model_params

        self.df_trainset = None
        self.df_testset = None

        if self.repeats is None:
            self.seed_list = [33]
        else:
            # generate a random seed list based on the number of repeats
            np.random.seed(33)
            self.seed_list = np.random.randint(0, 10000, self.repeats)

        if self.ml_model == 'Lasso':
            self.ml_model = linear_model.Lasso()
            self.model_name = 'lasso'

        if self.ml_model == 'ElasticNet':
            self.ml_model = linear_model.ElasticNet()
            self.model_name = 'elasticnet'

        if self.ml_model == 'KernelRidge':
            self.ml_model = KernelRidge()
            self.model_name = 'kernelridge'


    def train_model(self):
        """Run the model.

        Args:

        Returns:
            scores (dataframe): The dataframe of scores.
            model (model): The model.
        """
        # run the model
        if self.confounds_list is None:
            scores, model = run_cross_validation(
            X=self.X_list, y=self.y_list[0], data=self.df_trainset, preprocess_X='zscore',
            problem_type='regression', model=self.ml_model, cv=self.cv_folds, return_estimator='all', return_train_score=True,
            model_params=self.model_params, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'], seed=33)

        else:
            scores, model = run_cross_validation(
            X=self.X_list, y=self.y_list[0], data=self.df_trainset, confounds=self.confounds_list , preprocess_X=['zscore', 'remove_confounds'],
            problem_type='regression', model=self.ml_model, cv=self.cv_folds, return_estimator='all', return_train_score=True,
            model_params=self.model_params, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'], seed=33)

        return scores, model


    def test_model(self, model):
        """Test the model.
        Metrics: MAE, MSE, R2, pearson correlation.
        Model significance: p-value, F-statistic

        Args:

        Returns: List of metrics and model significance

        """
        # get the testset
        X_test = self.df_testset[self.X_list]
        y_test = self.df_testset[self.y_list[0]]

        # get the prediction
        y_pred = model.predict(X_test)

        # get the metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # print('y_test y_pred shape:\n', y_test.shape, y_pred.shape)

        # if y_test's values are all the same
        if len(np.unique(y_test)) == 1:
            print('X_test:\n', X_test)
            print('y_test:\n', y_test)
            print('model coef:\n', model[self.model_name].coef_)
            print('y_pred:\n', y_pred)

        pearson_corr = np.corrcoef(y_test, y_pred)[1, 0]
        if np.isnan(pearson_corr):
            print('X_test:\n', X_test)
            print('y_test:\n', y_test)
            print('model coef:\n', model[self.model_name].coef_)
            print('y_pred:\n', y_pred)

        # get the variables significance
        f_stats_vars, p_value_vars = f_regression(X_test, y_test)

        # explained sum of squares
        ESS = np.sum((y_pred - np.mean(y_test)) ** 2)

        # number of regression parameters
        if self.model_name == 'svm':
            p = len(model[self.model_name].coef_[0])
        else:
            p = len(model[self.model_name].coef_)
        # df_1, numerator degrees of freedom
        df_1 = p
        # nubmer of observations
        n = len(y_test)
        # degree of freedom for error
        df_2 = n - p - 1

        # residual sum of squares
        RSS = np.sum((y_test - y_pred) ** 2)

        # get the model significance
        f_stats_model, p_value_model = self.overall_regression_f(ESS, RSS, df_1, df_2)

        results_list = [mae, mse, r2, pearson_corr, f_stats_model, p_value_model]

        return results_list, f_stats_vars, p_value_vars


    def repeat_model(self):
        """Repeat the model.

        Return a list of metrics and model significance, which includes:
        1. mean_mae +- std, range_mae
        2. mean_mse +- std, range_mse
        3. mean_r2 +- std, range_r2
        4. mean_pearson_corr +- std, range_pearson_corr
        5. mean_f_statistic +- std, range_f_statistic
        6. mean_p_value +- std, range_p_value
        7. number of significant repeats, based on p_value < 0.05

        Args:

        Returns:

        """
        # initialize the results list
        results_model_list = []
        vars_weights_list = []
        f_stats_vars_list = []
        p_value_vars_list = []

        # outlier removal
        # TODO: maybe need to be inside the repeat loop
        df_dataset_cld_out = self.outliers_detect_remove()

        # repeat the model
        for seed in self.seed_list:
            # split the dataset into trainset and testset
            self.df_trainset, self.df_testset = train_test_split(df_dataset_cld_out, test_size=0.2, random_state=seed)

            if self.ml_model == 'Polynomial':
                self.ml_model = 'linreg'
                self.model_name = 'linreg'
                poly = PolynomialFeatures(degree=2)
                X_train = poly.fit_transform(self.df_trainset[self.X_list])
                X_test = poly.fit_transform(self.df_testset[self.X_list])
                y_train = self.df_trainset[self.y_list]
                y_test = self.df_testset[self.y_list]

                X_list_poly = poly.get_feature_names_out().tolist()
                X_list_poly = [x.replace('^2', '_2') for x in X_list_poly]

                df_trainset_poly = pd.DataFrame(X_train, columns=X_list_poly)
                df_testset_poly = pd.DataFrame(X_test, columns=X_list_poly)
                df_trainset_poly = df_trainset_poly.assign(**{self.y_list[0]: y_train.values})
                df_testset_poly = df_testset_poly.assign(**{self.y_list[0]: y_test.values})

                self.X_list = X_list_poly
                self.df_trainset = df_trainset_poly
                self.df_testset = df_testset_poly

                scores, model = self.train_model()

            else:
                # run the model
                scores, model = self.train_model()

            # test the model
            results_model, f_stats_vars, p_value_vars = self.test_model(model)

            # get the feature importance
            if self.model_name == 'svm':
                vars_weights = model[self.model_name].coef_[0]
            else:
                vars_weights = model[self.model_name].coef_

            # append the results
            results_model_list.append(results_model)
            vars_weights_list.append(vars_weights)
            f_stats_vars_list.append(f_stats_vars)
            p_value_vars_list.append(p_value_vars)

            # print how many repeats finished
            print('Finished {} repeats'.format(len(results_model_list)))

        model_metrics_list = self.model_metrics_results(results_model_list)
        vars_metric_df = self.vars_significance_results(vars_weights_list, f_stats_vars_list, p_value_vars_list)

        return model_metrics_list, vars_metric_df


    def model_metrics_results(self, results_list):
        """Calculate the average and range of the results.
        Results order: mae, mse, r2, pearson_corr, f_stats_model, p_value_model, number of significant models

        """
        results_model_array = np.array(results_list)

        mean_mae, sd_mae, max_mae, min_mae = self.mean_sd_range(results_model_array[:, 0])
        mean_mse, sd_mse, max_mse, min_mse = self.mean_sd_range(results_model_array[:, 1])
        mean_r2, sd_r2, max_r2, min_r2 = self.mean_sd_range(results_model_array[:, 2])
        mean_pearson_corr, sd_pearson_corr, max_pearson_corr, min_pearson_corr = self.mean_sd_range(
            results_model_array[:, 3])
        mean_f_stats_model, sd_f_stats_model, max_f_stats_model, min_f_stats_model = self.mean_sd_range(
            results_model_array[:, 4])
        mean_p_value_model, sd_p_value_model, max_p_value_model, min_p_value_model = self.mean_sd_range(
            results_model_array[:, 5])

        # count the number of significant models
        num_sig_model = 0
        for i in range(len(results_model_array[:, 5])):
            if results_model_array[i, 5] < 0.05:
                num_sig_model += 1

        model_metrics_list = [('%.3f' % mean_mae) + '±' + ('%.3f' % sd_mae),
                              '[' + ('%.3f' % min_mae) + ', ' + ('%.3f' % max_mae) + ']',
                              ('%.3f' % mean_mse) + '±' + ('%.3f' % sd_mse),
                              '[' + ('%.3f' % min_mse) + ', ' + ('%.3f' % max_mse) + ']',
                              ('%.3f' % mean_r2) + '±' + ('%.3f' % sd_r2),
                              '[' + ('%.3f' % min_r2) + ', ' + ('%.3f' % max_r2) + ']',
                              ('%.3f' % mean_pearson_corr) + '±' + ('%.3f' % sd_pearson_corr),
                              '[' + ('%.3f' % min_pearson_corr) + ', ' + ('%.3f' % max_pearson_corr) + ']',
                              ('%.3f' % mean_f_stats_model) + '±' + ('%.3f' % sd_f_stats_model),
                              '[' + ('%.3f' % min_f_stats_model) + ', ' + ('%.3f' % max_f_stats_model) + ']',
                              ('%.3f' % mean_p_value_model) + '±' + ('%.3f' % sd_p_value_model),
                              '[' + ('%.3f' % min_p_value_model) + ', ' + ('%.3f' % max_p_value_model) + ']',
                              num_sig_model]

        return model_metrics_list


    def vars_significance_results(self, vars_weights_list, f_stats_vars_list, p_value_vars_list):
        """Calculate the average and range of the p-value of the variables.

        """
        mean_weights, sd_weights, max_weights, min_weights = self.mean_sd_range(vars_weights_list, axis=0)
        mean_f_stats_vars, sd_f_stats_vars, max_f_stats_vars, min_f_stats_vars = self.mean_sd_range(f_stats_vars_list,
                                                                                               axis=0)
        mean_p_value_vars, sd_p_value_vars, max_p_value_vars, min_p_value_vars = self.mean_sd_range(p_value_vars_list,
                                                                                               axis=0)

        # make a list of whether the variables are significant based on the p-value
        sig_vars_list = []
        for i in range(len(mean_p_value_vars)):
            if mean_p_value_vars[i] < 0.05:
                sig_vars_list.append('Yes')
            else:
                sig_vars_list.append('No')

        # make a list of how many times the variables are significant based on the p-value
        num_sig_vars_list = []
        for i in range(len(mean_p_value_vars)):
            num_sig_vars = 0
            for j in range(len(p_value_vars_list)):
                if p_value_vars_list[j][i] < 0.05:
                    num_sig_vars += 1
            num_sig_vars_list.append(num_sig_vars)

        vars_list = self.X_list
        mean_weights_sd = [('%.3f' % mean_weights[i]) + '±' + ('%.3f' % sd_weights[i]) for i in
                           range(len(mean_weights))]
        range_weights = ['[' + ('%.3f' % min_weights[i]) + ', ' + ('%.3f' % max_weights[i]) + ']' for i in
                         range(len(min_weights))]
        mean_f_stats_vars_sd = [('%.3f' % mean_f_stats_vars[i]) + '±' + ('%.3f' % sd_f_stats_vars[i]) for i in
                                range(len(mean_f_stats_vars))]
        range_f_stats_vars = ['[' + ('%.3f' % min_f_stats_vars[i]) + ', ' + ('%.3f' % max_f_stats_vars[i]) + ']' for i
                              in
                              range(len(min_f_stats_vars))]
        mean_p_value_vars_sd = [('%.3f' % mean_p_value_vars[i]) + '±' + ('%.3f' % sd_p_value_vars[i]) for i in
                                range(len(mean_p_value_vars))]
        range_p_value_vars = ['[' + ('%.3f' % min_p_value_vars[i]) + ', ' + ('%.3f' % max_p_value_vars[i]) + ']' for i
                              in
                              range(len(min_p_value_vars))]

        vars_results_df = pd.DataFrame(
            {'vars': vars_list, 'mean_weights±sd_weights': mean_weights_sd, 'range_weights': range_weights,
             'mean_f_stats_vars±sd_f_stats_vars': mean_f_stats_vars_sd, 'range_f_stats_vars': range_f_stats_vars,
             'mean_p_value_vars±sd_p_value_vars': mean_p_value_vars_sd, 'range_p_value_vars': range_p_value_vars,
             'sig_vars': sig_vars_list, 'sig_nums': num_sig_vars_list})

        return vars_results_df


    def mean_sd_range(self, results_list, axis=None):

        if axis is None:
            mean = np.mean(results_list)
            sd = np.std(results_list)
            max = np.max(results_list)
            min = np.min(results_list)

        elif axis == 0:
            mean = np.mean(results_list, axis=0)
            sd = np.std(results_list, axis=0)
            max = np.max(results_list, axis=0)
            min = np.min(results_list, axis=0)

        elif axis == 1:
            mean = np.mean(results_list, axis=1)
            sd = np.std(results_list, axis=1)
            max = np.max(results_list, axis=1)
            min = np.min(results_list, axis=1)

        return mean, sd, max, min


    def outliers_detect_remove(self):
        """Detect and remove outliers.
        Implemented by using the IsolationForest algorithm.

        Returns:
            df_dataset_cld_outliers_removed: the dataset with outliers removed.
        """
        # outlier detection by isolation forest
        clf = IsolationForest(n_estimators=10, warm_start=True)
        clf.fit(self.df_dataset[self.X_list + self.y_list])

        outliers = clf.predict(self.df_dataset[self.X_list + self.y_list])
        # count the number of outliers
        print('Number of outliers:', np.count_nonzero(outliers == -1))
        # remove outliers
        df_dataset_cld_out = self.df_dataset.iloc[outliers == 1, :]

        return df_dataset_cld_out


    def overall_regression_f(self, ESS, RSS, df_1, df_2):
        """Calculate the overall F-test for the model.

        Args:

        Returns: F value, p-value
        """
        F = (ESS / df_1) / (RSS / df_2)
        p_value = st.f.sf(F, df_1, df_2)

        return F, p_value