import os
import pandas as pd
import time
import multiprocessing
from joblib import dump, load

import numpy as np
import scipy
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

import julearn
from julearn import run_cross_validation
from julearn.utils import configure_logging

from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
# from sklearn.inspection import permutation_importance

# %%
def repeat_different_x_input_SVM_rbf(combinations_name, x_list, y_list, data_input, cv_data_splits,
                                     model_params, num_cores_use, results_path, confounds_list=None):
    for y_target in y_list:
        start_time = time.time()

        print('\nSVM-rbf for %s to predict %s is starting.\n' % (combinations_name, y_target))

        if confounds_list is None:
            scores_, model_ = run_cross_validation(
                X=x_list, y=y_target, data=data_input, problem_type='regression', model='svm', preprocess_X='zscore',
                cv=cv_data_splits, return_estimator='all', return_train_score=True, model_params=model_params,
                scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2', 'r2_corr'],
                seed=200, n_jobs=num_cores_use)

        if confounds_list is not None:
            scores_, model_ = run_cross_validation(
                X=x_list, y=y_target, data=data_input, confounds=confounds_list, problem_type='regression',
                model='svm', preprocess_X=['zscore', 'remove_confound'], preprocess_confounds='zscore',
                cv=cv_data_splits, return_estimator='all', return_train_score=True, model_params=model_params,
                scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2', 'r2_corr'], seed=200,
                n_jobs=num_cores_use)

        print('\nSVM-rbf for %s to predict %s is done.\n' % (combinations_name, y_target))

        # save model
        dump(model_, results_path + 'model_SVM_rbf_' + y_target + '_' + combinations_name + '.joblib')

        # print mean and std of scores
        print('Mean of test_neg_mean_absolute_error: {} ± {}'.format(np.mean(scores_["test_neg_mean_absolute_error"]),
                                                                     np.std(scores_["test_neg_mean_absolute_error"])))
        print('Mean of test_neg_mean_squared_error: {} ± {}'.format(
            np.mean(scores_["test_neg_mean_squared_error"]),
            np.std(scores_["test_neg_mean_squared_error"])))
        print('Mean of test_r2: {} ± {}'.format(np.mean(scores_["test_r2"]),
                                                np.std(scores_["test_r2"])))
        print('Mean of test_r2_corr: {} ± {}'.format(np.mean(scores_["test_r2_corr"]),
                                                     np.std(scores_["test_r2_corr"])))

        scores_.to_csv(results_path + 'scores_model_SVM_rbf_' + combinations_name + '_' + y_target + '.csv', index=False)

        # scores line plot for each fold
        test_neg_mean_absolute_error_list = []
        test_neg_mean_squared_error_list = []
        test_r2_list = []
        test_r2_corr_list = []
        train_neg_mean_absolute_error_list = []
        train_neg_mean_squared_error_list = []
        train_r2_list = []
        train_r2_corr_list = []
        for i in range(len(scores_)):
            test_neg_mean_absolute_error_list.append(scores_["test_neg_mean_absolute_error"][i])
            test_neg_mean_squared_error_list.append(scores_["test_neg_mean_squared_error"][i])
            test_r2_list.append(scores_["test_r2"][i])
            test_r2_corr_list.append(scores_["test_r2_corr"][i])
            train_neg_mean_absolute_error_list.append(scores_["train_neg_mean_absolute_error"][i])
            train_neg_mean_squared_error_list.append(scores_["train_neg_mean_squared_error"][i])
            train_r2_list.append(scores_["train_r2"][i])
            train_r2_corr_list.append(scores_["train_r2_corr"][i])

        # make line plot
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        ax[0, 0].plot(test_neg_mean_absolute_error_list, 'o-')
        ax[0, 0].plot(train_neg_mean_absolute_error_list, 'x-')
        ax[0, 0].set_xlabel('Fold')
        ax[0, 0].set_ylabel('neg_mean_absolute_error')
        ax[0, 0].legend(['test', 'train'])
        ax[0, 0].set_title('neg_mean_absolute_error for each fold')

        ax[0, 1].plot(test_neg_mean_squared_error_list, 'o-')
        ax[0, 1].plot(train_neg_mean_squared_error_list, 'x-')
        ax[0, 1].set_xlabel('Fold')
        ax[0, 1].set_ylabel('neg_mean_squared_error')
        ax[0, 1].legend(['test', 'train'])
        ax[0, 1].set_title('neg_mean_squared_error for each fold')

        ax[1, 0].plot(test_r2_list, 'o-')
        ax[1, 0].plot(train_r2_list, 'x-')
        ax[1, 0].set_xlabel('Fold')
        ax[1, 0].set_ylabel('r2')
        ax[1, 0].legend(['test', 'train'])
        ax[1, 0].set_title('r2 for each fold')

        ax[1, 1].plot(test_r2_corr_list, 'o-')
        ax[1, 1].plot(train_r2_corr_list, 'x-')
        ax[1, 1].set_xlabel('Fold')
        ax[1, 1].set_ylabel('r2_corr')
        ax[1, 1].legend(['test', 'train'])
        ax[1, 1].set_title('r2_corr for each fold')

        # title for the whole figure
        fig.suptitle('SVM-rbf (%s) Scores for %s in SHIP' % (combinations_name, y_target))
        plt.tight_layout()
        plt.show()
        plt.savefig(results_path + 'figures/' + 'model_test_SHIP_%s_SVM_rbf_%s.png' % (combinations_name, y_target))
        plt.close()

        # print the best estimator
        print('\nBest estimator for %s to predict %s' % (combinations_name, y_target))
        print(model_.best_params_)
        print('\n')
        # hyperparameter for each fold
        C_list = []
        gamma_list = []
        for i in range(len(scores_)):
            C_list.append(scores_["estimator"][i].best_estimator_['svm'].C)
            gamma_list.append(scores_["estimator"][i].best_estimator_['svm'].gamma)

        # make line plot
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(C_list, 'o-')
        ax[0].set_xlabel('Fold')
        ax[0].set_ylabel('C')
        ax[0].set_title('C for each fold')
        ax[1].plot(gamma_list, 'o-')
        ax[1].set_xlabel('Fold')
        ax[1].set_ylabel('gamma')
        ax[1].set_title('gamma for each fold')
        # title for the whole figure
        fig.suptitle('SVM-rbf (%s) hyperparameter for %s in SHIP' % (combinations_name, y_target))
        plt.tight_layout()
        plt.show()
        plt.savefig(results_path + 'figures/' + 'hyperparameter_SHIP_%s_SVM_rbf_%s.png' % (combinations_name, y_target))
        plt.close()

        # """
        # Save the feature importance of the best estimator
        # """
        # r_multi_r2 = []
        #
        # for i in range(len(cv_data_splits)):
        #     validation_index = cv_data_splits[i][1]
        #     X_val = data_input.iloc[validation_index, :][permutation_x_list]
        #     y_val = data_input.iloc[validation_index, :][y_target]
        #     r_multi = permutation_importance(model_, X_val, y_val, n_repeats=permutation_repeats,
        #                                      random_state=33, n_jobs=num_cores_use, scoring=['r2'])
        #
        #     r_multi_r2.append(r_multi['r2'].importances_mean)
        #
        # # for r_multi_mae, make a dataframe, column is the feature, row is the permutation importance
        # r_multi_r2_df = pd.DataFrame(r_multi_r2, columns=permutation_x_list)
        #
        # # save the permutation importance to the results folder
        # r_multi_r2_df.to_csv(
        #     results_path + 'permutation_importance_r2_' + 'model_SVM_rbf_'+ combinations_name + '_' + y_target + '.csv',
        #     index=False)

        finish_time = time.time()
        print('\nTime used for model_SVM_rbf_%s for %s in SHIP: %s\n' % (combinations_name, y_target,
                                                                         finish_time - start_time))