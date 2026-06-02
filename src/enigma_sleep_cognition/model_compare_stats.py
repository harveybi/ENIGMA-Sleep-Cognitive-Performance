import os

import pandas as pd
import numpy as np
import scipy
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

import statannotations
from statannotations.Annotator import Annotator

# %%
def make_boxplot_annotations(df_MAE, df_MSE, df_R2, df_corr, title, hue=None, hue_order=None):
    fig, axes = plt.subplots(2, 2, figsize=(30, 15))
    x = 'Feature'
    y = 'values'

    # Check if dataframes have a 'Train/Test' column
    has_train_test_col = all('Train/Test' in df.columns for df in [df_MAE, df_MSE, df_R2, df_corr])

    # Function to generate pairs from the data
    def generate_pairs(data):
        models = data['Feature'].unique()
        pairs = []
        for i in range(len(models) - 1):
            for j in range(i + 1, len(models)):
                pairs.append((models[i], models[j]))
        return pairs

    if hue is not None:
        order = df_MAE[x].unique()
        dataframes = [df_MAE, df_MSE, df_R2, df_corr]
        for idx, ax in enumerate(axes.flat):
            sns.boxplot(data=dataframes[idx], x=x, y=y, order=order, hue=hue, hue_order=hue_order,
                        ax=ax).legend_.remove()
            pairs = generate_pairs(dataframes[idx][dataframes[idx]['Train/Test'] == 'Test'])
            annot = Annotator(ax, pairs, data=dataframes[idx], x=x, y=y, order=order, hue=hue, hue_order=hue_order,
                              hide_non_significant=True)
            annot.configure(test='t-test_paired', text_format='star', loc='inside', comparisons_correction="Bonferroni",
                            verbose=2)
            annot.apply_test()
            ax, _ = annot.annotate()

    else:
        dataframes = [df_MAE, df_MSE, df_R2, df_corr]
        for idx, ax in enumerate(axes.flat):
            if has_train_test_col:
                sns.boxplot(data=dataframes[idx][dataframes[idx]['Train/Test'] == 'Test'], x=x, y=y, ax=ax)
            else:
                sns.boxplot(data=dataframes[idx], x=x, y=y, ax=ax)
            pairs = generate_pairs(dataframes[idx])
            annot = Annotator(ax, pairs, data=dataframes[idx], x=x, y=y, hide_non_significant=True)
            annot.configure(test='t-test_paired', text_format='star', loc='inside', comparisons_correction="Bonferroni",
                            verbose=2)
            annot.apply_test()
            ax, _ = annot.annotate()

    # change x-axis labels 45 degree
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 1].tick_params(axis='x', rotation=45)

    axes[0, 0].set_title('MAE')
    axes[0, 1].set_title('RMSE')
    axes[1, 0].set_title('r2')
    axes[1, 1].set_title('Correlation')

    # change y-axis labels
    axes[0, 0].set_ylabel('Mean Absolute Error (MAE)')
    axes[0, 1].set_ylabel('Root Mean Squared Error (RMSE))')
    axes[1, 0].set_ylabel('Coefficient of Determination (R2)')
    axes[1, 1].set_ylabel('Correlation Coefficient (r)')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # set a main title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()


def make_boxplot_no_annotations(df_MAE, df_MSE, df_R2, df_corr, title, hue=None, hue_order=None):
    fig, axes = plt.subplots(2, 2, figsize=(30, 15))
    x = 'Feature'
    y = 'values'

    # Check if dataframes have a 'Train/Test' column
    has_train_test_col = all('Train/Test' in df.columns for df in [df_MAE, df_MSE, df_R2, df_corr])

    if hue is not None:
        order = df_MAE[x].unique()
        sns.boxplot(data=df_MAE, x=x, y=y, order=order, hue=hue, hue_order=hue_order, ax=axes[0, 0]).legend_.remove()
        sns.boxplot(data=df_MSE, x=x, y=y, order=order, hue=hue, hue_order=hue_order, ax=axes[0, 1]).legend_.remove()
        sns.boxplot(data=df_R2, x=x, y=y, order=order, hue=hue, hue_order=hue_order, ax=axes[1, 0]).legend_.remove()
        sns.boxplot(data=df_corr, x=x, y=y, order=order, hue=hue, hue_order=hue_order, ax=axes[1, 1]).legend_.remove()

    else:
        if has_train_test_col:
            # only plot the Train/Test = Test
            sns.boxplot(data=df_MAE[df_MAE['Train/Test'] == 'Test'], x=x, y=y, ax=axes[0, 0])
            sns.boxplot(data=df_MSE[df_MSE['Train/Test'] == 'Test'], x=x, y=y, ax=axes[0, 1])
            sns.boxplot(data=df_R2[df_R2['Train/Test'] == 'Test'], x=x, y=y, ax=axes[1, 0])
            sns.boxplot(data=df_corr[df_corr['Train/Test'] == 'Test'], x=x, y=y, ax=axes[1, 1])
        else:
            sns.boxplot(data=df_MAE, x=x, y=y, ax=axes[0, 0])
            sns.boxplot(data=df_MSE, x=x, y=y, ax=axes[0, 1])
            sns.boxplot(data=df_R2, x=x, y=y, ax=axes[1, 0])
            sns.boxplot(data=df_corr, x=x, y=y, ax=axes[1, 1])

    # change x-axis labels 45 degree
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 1].tick_params(axis='x', rotation=45)

    axes[0, 0].set_title('MAE')
    axes[0, 1].set_title('RMSE')
    axes[1, 0].set_title('r2')
    axes[1, 1].set_title('Correlation')

    # change y-axis labels
    axes[0, 0].set_ylabel('Mean Absolute Error (MAE)')
    axes[0, 1].set_ylabel('Root Mean Squared Error (RMSE))')
    axes[1, 0].set_ylabel('Coefficient of Determination (R2)')
    axes[1, 1].set_ylabel('Correlation Coefficient (r)')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # set a main title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()


def make_df_metrics_train_test_performance(dfs, metric_name):
    df_metrics = pd.DataFrame()

    # Check if any dataframe in 'dfs' contains columns with "train" or "test"
    contains_train_or_test = any(any('train' in col or 'test' in col for col in df.columns) for df in dfs.values())

    if not contains_train_or_test:
        for label, df in dfs.items():
            temp_df_metrics = pd.DataFrame({'values': df[metric_name].values,
                                    'Feature': label,
                                    'Model': df['Best_Model'].values})
            df_metrics = pd.concat([df_metrics, temp_df_metrics], axis=0)
            # then jump out of the function
        return df_metrics

    # if there are train or test word in dataframe column name, execute the following
    train_metric_name = 'train_' + metric_name
    test_metric_name = 'test_' + metric_name

    for label, df in dfs.items():
        model, feature = '_'.join(label.split("_")[:2]), '_'.join(label.split("_")[2:])

        if train_metric_name in df.columns:
            temp_df_train = pd.DataFrame({'values': df[train_metric_name].values,
                                          'Model': label,
                                          'Feature': feature,
                                          'Train/Test': 'Train'})

            df_metrics = pd.concat([df_metrics, temp_df_train], axis=0)

        if test_metric_name in df.columns:
            temp_df_test = pd.DataFrame({'values': df[test_metric_name].values,
                                        'Model': model,
                                        'Feature': feature,
                                        'Train/Test': 'Test'})
            df_metrics = pd.concat([df_metrics, temp_df_test], axis=0)

    return df_metrics



def make_df_metrics_test_performance_compare_features(dfs, metric_name):
    df_metrics = pd.DataFrame()
    # test_metric_name = 'test_' + metric_name
    test_metric_name = metric_name

    for label, df in dfs.items():
        temp_df = pd.DataFrame({'values': df[test_metric_name].values,
                                'Feature': label,
                                'Train/Test': 'Test'})
        df_metrics = pd.concat([df_metrics, temp_df], axis=0, ignore_index=True)

    return df_metrics


def make_df_metrics_AutoGluon_performance(dfs, metric_name):
    return

def read_all_csv_from_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise ValueError("The specified folder does not exist.")

    # List all files in the directory
    files = os.listdir(folder_path)

    # Filter for only CSV files
    csv_files = [f for f in files if f.endswith('.csv')]

    # Create a dictionary to store each dataframe with filename (without extension) as the key
    dataframes = {}

    for csv_file in csv_files:
        # Remove .csv from the filename to use as dataframe name
        df_name = csv_file[:-4]
        # Read the CSV file into a dataframe
        dataframes[df_name] = pd.read_csv(os.path.join(folder_path, csv_file))

    return dataframes