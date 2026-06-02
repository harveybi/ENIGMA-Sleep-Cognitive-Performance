import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer


# %%
def add_age_groups(dataframe):
    # Function to divide subjects into three age groups
    def divide_age_groups(age_series):
        # Finding the quantiles to divide the age into three approximately equal groups
        age_quantiles = age_series.quantile([1 / 3, 2 / 3])
        age_groups = []

        # Dividing the age into groups based on the quantiles
        for age in age_series:
            if age <= age_quantiles[1 / 3]:
                age_groups.append(f'{int(age_series.min())}-{int(age_quantiles[1 / 3])}')
            elif age <= age_quantiles[2 / 3]:
                age_groups.append(f'{int(age_quantiles[1 / 3]) + 1}-{int(age_quantiles[2 / 3])}')
            else:
                age_groups.append(f'{int(age_quantiles[2 / 3]) + 1}-{int(age_series.max())}')

        return age_groups

    # Applying the function to the 'Age_at_Scan' column
    dataframe['Age_Group'] = divide_age_groups(dataframe['Age_at_Scan'])
    # dataframe.loc[:, 'Age_Group'] = divide_age_groups(dataframe['Age_at_Scan'])

    # Print unique age groups
    unique_age_groups = dataframe['Age_Group'].unique()
    print("Age Groups:", unique_age_groups)

    # Reordering columns to have 'Age_Group' after 'Age_at_Scan'
    columns_order = dataframe.columns.tolist()
    columns_order.insert(columns_order.index('Age_at_Scan') + 1, columns_order.pop(columns_order.index('Age_Group')))
    dataframe = dataframe[columns_order]

    return dataframe


def add_any_age_groups(dataframe, group_nums=3):
    # Function to divide subjects into specified number of age groups
    def divide_age_groups(age_series, group_nums):
        # Finding the quantiles to divide the age into group_nums groups
        quantiles = age_series.quantile([i / group_nums for i in range(1, group_nums)]).tolist()
        age_groups = []

        # Dividing the age into groups based on the quantiles
        for age in age_series:
            added = False
            for i in range(len(quantiles)):
                if age <= quantiles[i]:
                    if i == 0:
                        age_groups.append(f'{int(age_series.min())}-{int(quantiles[i])}')
                    else:
                        age_groups.append(f'{int(quantiles[i - 1]) + 1}-{int(quantiles[i])}')
                    added = True
                    break
            if not added:
                age_groups.append(f'{int(quantiles[-1]) + 1}-{int(age_series.max())}')

        return age_groups

    # Applying the function to the 'Age_at_Scan' column
    dataframe['Age_Group'] = divide_age_groups(dataframe['Age_at_Scan'], group_nums)

    # Print unique age groups
    unique_age_groups = dataframe['Age_Group'].unique()
    print("Age Groups:", unique_age_groups)

    # Reordering columns to have 'Age_Group' after 'Age_at_Scan'
    columns_order = dataframe.columns.tolist()
    columns_order.insert(columns_order.index('Age_at_Scan') + 1, columns_order.pop(columns_order.index('Age_Group')))
    dataframe = dataframe[columns_order]

    return dataframe


def add_groups_age_sex(dataframe):
    # Create a temporary column to map 'SEX' without modifying the original
    dataframe['_temp_sex'] = dataframe['SEX'].map({1: 'M', 2: 'F'})

    # Create the new combined column
    dataframe['Group_Age_SEX'] = dataframe['Age_Group'].astype(str) + "_" + dataframe['_temp_sex']

    # Drop the temporary column
    dataframe.drop('_temp_sex', axis=1, inplace=True)

    # Reorder the columns to place 'Age_Sex_Group' after 'SEX'
    cols = dataframe.columns.tolist()
    cols.insert(cols.index('SEX') + 1, cols.pop(cols.index('Group_Age_SEX')))
    dataframe = dataframe[cols]

    # Print unique groups
    unique_age_groups = dataframe['Group_Age_SEX'].unique()
    print("Age and SEX Groups:", unique_age_groups)

    return dataframe


def clean_missing_data(df, variable_list):
    # Calculate missing data
    missing_data = df[variable_list].isnull().sum(axis=1)
    missing_data_count = (missing_data > 0).sum()

    print('Total number of subjects: {}'.format(df.shape[0]))
    print('Number of subjects with missing data: {}'.format(missing_data_count))

    # Remove subjects with missing data in necessary features
    df_cld = df.dropna(subset=variable_list)
    # Print how many subjects in total now
    print('Total number of subjects after removing missing data: {}'.format(df_cld.shape[0]))

    return df_cld


def convert_units(df, sleep_dur_cols, sleep_eff_cols):
    for col in sleep_dur_cols:
        if df[col].max() > 24:
            df[col] = df[col] / 60
            print(f'{col} converted to hours')
        else:
            print(f'{col} already in hours')

    for col in sleep_eff_cols:
        if df[col].max() > 2:
            df[col] = df[col] / 100
            print(f'{col} converted to percentage')
        else:
            print(f'{col} already in percentage')

    return df


def dataset_loader(file_path):
    """Load dataset for each site.

    For each site, split data into demographics and imaging data.

    Args:
        file_path (str): Path to the data file.

    Return:
        df_site_demo, df_site_imaging (DataFrame): DataFrame with demographic data.
    """

    # Tell whether the file is csv of Excel and read data from the file
    if file_path.endswith('.csv'):
        df_site = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df_site = pd.read_excel(file_path)

    # With 'Notes' as the dividing line, the column before the dividing line is demographics data, and the column after the dividing line is imaging data
    df_site_demo = df_site.iloc[:, :df_site.columns.get_loc('Notes')]
    df_site_imaging = df_site.iloc[:, df_site.columns.get_loc('Notes'):]

    df_sit_sub_id = df_site_demo['Sub_ID']
    df_site_imaging.rename(columns={'Notes': 'Sub_ID'}, inplace=True)
    df_site_imaging['Sub_ID'] = df_sit_sub_id

    return df_site_demo, df_site_imaging


def demo_stats(site_name, site_df, result_path):
    """Calculate statistics for demographic data.

    For continuous data, calculate the mean, std, and missing value number of each demographic variable.
    For discrete data, calculate the percentage of one category and the percentage of missing value of each demographic variable.

    Args:
        site_name (str): Site name.
        site_df (DataFrame): DataFrame with demographic data.

    Return:
        Print out the statistics of demographic data. Save the statistics to a csv file.
    """

    fw = open(result_path + site_name + "_data_stats.txt", 'w')

    print('\n')
    fw.write('\n')
    print('Checking', site_name, 'demographic data...')
    fw.write('Checking ' + site_name + ' demographic data...\n')

    categorical_var = ['SEX', 'Education', 'Chronotype', 'Hypertension', 'Alcohol', 'Smoking', 'Marital_Status',
                       'APOE4']
    for i in range(len(site_df.columns)):
        if site_df.columns[i] in categorical_var:
            if np.shape(site_df.groupby(site_df.columns[i])[site_df.columns[i]].count())[0] == 2:
                y, n = site_df.groupby(site_df.columns[i])[site_df.columns[i]].count()
                m = site_df[site_df.columns[i]].isnull().sum(axis=0)
                print('\n')
                fw.write('\n')
                print(site_df.columns[i], ':')
                fw.write(site_df.columns[i] + ':\n')
                print(site_df.columns[i], '_1 number is:', y)
                fw.write(site_df.columns[i] + '_1 number is: ' + str(y) + '\n')
                print(site_df.columns[i], '_1 percent is:', y / (y + n + m))
                fw.write(site_df.columns[i] + '_1 percent is: ' + str(y / (y + n + m)) + '\n')
                print(site_df.columns[i], '_2 number is:', n)
                fw.write(site_df.columns[i] + '_2 number is: ' + str(n) + '\n')
                print(site_df.columns[i], '_2 percent is:', n / (y + n + m))
                fw.write(site_df.columns[i] + '_2 percent is: ' + str(n / (y + n + m)) + '\n')
                print('Missing data number is:', m)
                fw.write('Missing data number is: ' + str(m) + '\n')
                print('Missing data percent is:', m / (y + n + m))
                fw.write('Missing data percent is: ' + str(m / (y + n + m)) + '\n')
            elif np.shape(site_df.groupby(site_df.columns[i])[site_df.columns[i]].count())[0] == 3:
                y, n, m = site_df.groupby(site_df.columns[i])[site_df.columns[i]].count()
                mm = site_df[site_df.columns[i]].isnull().sum(axis=0)
                print('\n')
                fw.write('\n')
                print(site_df.columns[i], ':')
                fw.write(site_df.columns[i] + ':\n')
                print(site_df.columns[i], '_1 number is:', y)
                fw.write(site_df.columns[i] + '_1 number is: ' + str(y) + '\n')
                print(site_df.columns[i], '_1 percent is:', y / (y + n + m + mm))
                fw.write(site_df.columns[i] + '_1 percent is: ' + str(y / (y + n + m + mm)) + '\n')
                print(site_df.columns[i], '_2 number is:', n)
                fw.write(site_df.columns[i] + '_2 number is: ' + str(n) + '\n')
                print(site_df.columns[i], '_2 percent is:', n / (y + n + m + mm))
                fw.write(site_df.columns[i] + '_2 percent is: ' + str(n / (y + n + m + mm)) + '\n')
                print(site_df.columns[i], '_3 number is:', m)
                fw.write(site_df.columns[i] + '_3 number is: ' + str(m) + '\n')
                print(site_df.columns[i], '_3 percent is:', m / (y + n + m + mm))
                fw.write(site_df.columns[i] + '_3 percent is: ' + str(m / (y + n + m + mm)) + '\n')
                print('Missing data number is:', mm)
                fw.write('Missing data number is: ' + str(mm) + '\n')
                print('Missing data percent is:', mm / (y + n + m + mm))
                fw.write('Missing data percent is: ' + str(mm / (y + n + m + mm)) + '\n')

        else:
            print('\n')
            fw.write('\n')
            print(site_df.columns[i], ':')
            fw.write(site_df.columns[i] + ':\n')
            m = site_df[site_df.columns[i]].isnull().sum(axis=0)
            print('Missing data number is:', m)
            fw.write('Missing data number is: ' + str(m) + '\n')
            print('Missing data percent is:', m / len(site_df))
            fw.write('Missing data percent is: ' + str(m / len(site_df)) + '\n')
            print(site_df[site_df.columns[i]].describe())
            fw.write(str(site_df[site_df.columns[i]].describe()) + '\n')

    fw.close()

    return None


def plot_corr_matrix(df, feature_list, site_name, results_path=None):
    # Calculate correlation matrices
    corrMatrix = df[feature_list].corr()
    corrMatrix_corrected = df[feature_list].rcorr(padjust='bonf')

    # Prepare labels for heatmap
    annot_label = np.array(corrMatrix_corrected.values.tolist())
    annot_label[annot_label == ''] = 'ns'

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corrMatrix, annot=annot_label, cmap=plt.cm.RdBu_r, fmt='', vmin=-1, vmax=1)
    plt.xticks(rotation=45)
    plt.title('Correlation matrix of phenotypic variables for ' + site_name)
    plt.tight_layout()

    # Save figure
    if results_path is not None:
        fig_path = results_path + site_name + '_Corr_Matrix.png'
        plt.show()
        plt.savefig(fig_path)
        print("Correlation matrix saved at: ", fig_path)
    else:
        plt.show()
        plt.close()


def plot_sleep_dur_distribute(site_name, site_df, save_path):
    """Plot the distribution of 'PSG_Sleep_Dur', 'Self_Sleep_Dur' for each dataset.

    Args:
        site_name (str): Site name.
        site_df (DataFrame): DataFrame with demographic data.

    Return:
        Plot the distribution of 'PSG_Sleep_Dur', 'Self_Sleep_Dur' for each dataset.
    """

    fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)
    if 'PSG_Sleep_Dur' in site_df.columns.to_list():
        PSG_Sleep_Dur = site_df['PSG_Sleep_Dur'].values
        if PSG_Sleep_Dur.mean() > 10:
            PSG_Sleep_Dur = PSG_Sleep_Dur / 60

        axs[0].hist(PSG_Sleep_Dur, bins=np.arange(1, 12, 0.5), color='C0', edgecolor='w', linewidth=0.5)
        axs[0].set_xlabel('PSG sleep duration (hours)')
        axs[0].set_ylabel('Count')
    else:
        print('PSG_Sleep_Dur is missing in', site_name)

        # Get Self_Sleep_Dur from dataframe site_df as numpy array
    if 'Self_Sleep_Dur' in site_df.columns.to_list():
        Self_Sleep_Dur = site_df['Self_Sleep_Dur'].values
        if Self_Sleep_Dur.mean() > 10:
            Self_Sleep_Dur = Self_Sleep_Dur / 60

        axs[1].hist(Self_Sleep_Dur, bins=np.arange(1, 12, 0.5), color='C1', edgecolor='w', linewidth=0.5)
        axs[1].set_xlabel('Self sleep duration (hours)')
        axs[1].set_ylabel('Count')
    else:
        print('Self_Sleep_Dur is missing in', site_name)

    fig.suptitle(site_name)
    fig.savefig(save_path + site_name + '_sleep_dur_distribute.png')
    print("Figure saved at: ", save_path + site_name + '_sleep_dur_distribute.png')
    plt.show()
    plt.close()

    return None


def plot_sleep_eff_distribute(site_name, site_df, save_path):
    """Plot the distribution of 'PSG_Sleep_Eff', 'Self_Sleep_Eff' for each dataset.

    Args:
        site_name (str): Site name.
        site_df (DataFrame): DataFrame with demographic data.

    Return:
        Plot the distribution of 'PSG_Sleep_Eff', 'Self_Sleep_Eff' for each dataset.
    """

    fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)
    if 'PSG_Sleep_Eff' in site_df.columns.to_list():
        PSG_Sleep_Eff = site_df['PSG_Sleep_Eff'].values

        # axs[0].hist(PSG_Sleep_Eff, bins=np.arange(30, 105, 5), color='C0', edgecolor='w', linewidth=0.5)
        axs[0].hist(PSG_Sleep_Eff, color='C0', edgecolor='w', linewidth=0.5)
        axs[0].set_xlabel('PSG sleep efficiency (%)')
        axs[0].set_ylabel('Count')
    else:
        print('PSG_Sleep_Eff is missing in', site_name)

    if 'Self_Sleep_Eff' in site_df.columns.to_list():
        Self_Sleep_Eff = site_df['Self_Sleep_Eff'].values

        # axs[1].hist(Self_Sleep_Eff, bins=np.arange(30, 105, 5), color='C1', edgecolor='w', linewidth=0.5)
        axs[1].hist(Self_Sleep_Eff, color='C1', edgecolor='w', linewidth=0.5)
        axs[1].set_xlabel('Self sleep efficiency (%)')
        axs[1].set_ylabel('Count')
    else:
        print('Self_Sleep_Eff is missing in', site_name)

    fig.suptitle(site_name)
    fig.savefig(save_path + site_name + '_sleep_eff_distribute.png')
    print("Figure saved at: ", save_path + site_name + '_sleep_eff_distribute.png')
    plt.show()
    plt.close()

    return None


def plot_variable_distributions(df, variable_list, site_name, save_path):
    """Plot the distribution of given variables for the dataset.

    Args:
        df (DataFrame): DataFrame with demographic data.
        variable_list (list): List of variable names to plot.
        site_name (str): Site name.
        save_path (str): Path to save the plots.

    Returns:
        None
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']  # A list of colors

    # TODO: Maybe try this?
    # for feature in X_list:
    #     sns.set_style("whitegrid")
    #     plt.figure(figsize=(8, 6))
    #     sns.histplot(df_SHIP_ml_cld_out[feature], kde=True)
    #     plt.title(f"Distribution of {feature}")
    #     plt.tight_layout()
    #     plt.show()

    for i, variable in enumerate(variable_list):
        if variable in df.columns.to_list():
            data = df[variable].values

            plt.figure(tight_layout=True)
            sns.histplot(data, color=colors[i % len(colors)], edgecolor='w',
                         linewidth=0.5)  # Choose color cyclically from the list
            plt.xlabel(variable)

            fig_path = save_path + site_name + '_' + variable + '_distribution.png'
            plt.savefig(fig_path)
            plt.show()
            plt.close()

            print("Figure saved at: ", fig_path)
        else:
            print(variable, 'is missing in', site_name)


def plot_variable_relationship(df, variable_list, y_variable, poly_order, title, save_path, color_map=None,
                               fontsize=14):
    n_cols = len(variable_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 10), sharey=True, tight_layout=True)

    for i, variable in enumerate(variable_list):
        if color_map is None:
            sns.regplot(x=variable, y=y_variable, data=df, order=poly_order, truncate=False, line_kws={'color': 'red'},
                        marker='x', ax=axes[i])
        else:
            for age_group, color in color_map.items():
                age_group_data = df[df['Age_Group'] == age_group]
                if age_group_data.empty:
                    continue
                sns.regplot(x=variable, y=y_variable, data=age_group_data, order=poly_order, truncate=False,
                            line_kws={'color': color}, marker='x', ax=axes[i], label=f"Age Group: {age_group}")
            axes[i].legend(fontsize=fontsize - 2)

        # Add x-labels and configure ticks for all subplots
        axes[i].set_xlabel(variable, fontsize=fontsize)
        axes[i].tick_params(axis='both', which='major', labelsize=fontsize - 2)

        # Remove y-labels for all but the first subplot
        if i != 0:
            axes[i].set_ylabel('')

    # Add a single y-label to the first subplot
    axes[0].set_ylabel(y_variable, fontsize=fontsize)

    plt.suptitle(title, fontsize=fontsize + 2)
    plt.show()
    fig_path = f"{save_path}{title}.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Figure saved at: {fig_path}")


def plot_violin_plots(df, variable_list, fig_title, save_path, group_column=None, group_mapping=None, figsize=None,
                      fontsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)

    if group_column is None or group_mapping is None:
        if len(variable_list) != 2:
            raise ValueError("When not comparing groups, variable_list must contain exactly two variables to compare.")
        else:
            ax = sns.violinplot(data=df[variable_list])
            pairs = [(variable_list[0], variable_list[1])]
            annot = Annotator(ax, pairs, data=df[variable_list])
            annot.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction="Bonferroni",
                            verbose=2)
            annot.apply_test().annotate(line_offset_to_group=0.4)

            if fontsize is not None:
                plt.title(fig_title, fontsize=fontsize)
                plt.ylabel('Sleep duration (hours)', fontsize=fontsize)
                plt.xticks(fontsize=fontsize - 2)
                plt.yticks(fontsize=fontsize - 2)
            else:
                plt.title(fig_title)
                plt.ylabel('Sleep duration (hours)')

            plt.tight_layout()
            plt.show()
            plt.savefig(save_path)
            print("Figure saved at: ", save_path)
            plt.close()
    else:
        temp_df = df.copy()
        temp_df[group_column] = temp_df[group_column].map(group_mapping)
        for variable in variable_list:
            ax = sns.violinplot(x=group_column, y=variable, data=temp_df)
            annot = Annotator(ax, [(group_mapping[1], group_mapping[2])], data=temp_df, x=group_column, y=variable)
            annot.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction="Bonferroni",
                            verbose=2)
            annot.apply_test().annotate(line_offset_to_group=0.4)

            if fontsize is not None:
                plt.title(fig_title, fontsize=fontsize)
                plt.ylabel('Sleep duration (hours)', fontsize=fontsize)
                plt.xticks(fontsize=fontsize - 2)
                plt.yticks(fontsize=fontsize - 2)
            else:
                plt.title(fig_title)
                plt.ylabel('Sleep duration (hours)')
            plt.tight_layout()
            plt.show()
            plt.savefig(save_path)
            print("Figure saved at: ", save_path)
            plt.close()


def rm_missed_measure(df_site):
    """Delete columns that are completely empty.

    For the demographic tables, remove columns (measurements) which are 100% missing data.

    Args:
        df_site (DataFrame): DataFrame with demographic data.

    Return:
        df_site_all (DataFrame): DataFrame with demographic data which removed measurements with no contents.
    """

    df_site_headers = df_site.columns.values
    df_site_headers_no_missing = []
    for i in range(len(df_site_headers)):
        if df_site[df_site_headers[i]].isnull().sum() != len(df_site):
            df_site_headers_no_missing.append(df_site_headers[i])
    df_site_all = df_site[df_site_headers_no_missing]

    return df_site_all


def rm_outliers(df, variable_list, n_estimators=200, random_state=42):
    # # if there are missing data in the dataset, fill them with KNNImputer as df_imputed
    # if df[variable_list].isnull().sum().sum() > 0:
    #     print('\nNumber of subjects missing X and y data:', df[variable_list].isnull().sum().sum())
    #     print("Before outlier removal, will impute missing data with KNNImputer.")
    #     imputer = KNNImputer(n_neighbors=5)
    #     df_imputed = imputer.fit_transform(df[variable_list])
    #     df_imputed = pd.DataFrame(df_imputed, columns=variable_list)
    #
    #     # standardize the data
    #     scaler = StandardScaler()
    #     df_scl = scaler.fit_transform(df_imputed)
    # else:
    #     print('\nStart outlier removal.')
    #     # Standardize the data
    #     scaler = StandardScaler()
    #     df_scl = scaler.fit_transform(df[variable_list])

    print('\nStart outlier removal.')
    # Standardize the data
    scaler = StandardScaler()
    df_scl = scaler.fit_transform(df[variable_list])

    clf = IsolationForest(n_estimators=n_estimators, random_state=random_state)
    clf.fit(df_scl)

    # Predict the outliers in the dataset
    outliers = clf.predict(df_scl)

    # count the number of outliers
    print('Number of outliers:', np.count_nonzero(outliers == -1))
    # Remove outliers
    df_out = df.iloc[outliers == 1, :]

    # Check whether there are still missing data in X_list and y_list
    print('Total number of subjects now: {}'.format(df_out.shape[0]))
    print('Number of subject missing X and y data:', df_out[variable_list].isnull().sum().sum())

    return df_out


def evaluate_model_performance(model, scaler_Y, data, data_raw, X_input, y_column, model_name, feature_comb,
                               results_path):
    """
    Evaluate the performance of a model on the given data.
    """
    case_results_path = results_path + model_name + '/' + feature_comb + '/'
    if not os.path.exists(case_results_path):
        os.makedirs(case_results_path)

    # Making predictions
    predictions_scaled = model.predict(data[X_input])

    # Inverse transform to get predictions in original scale
    predictions_scaled_reshaped = np.array(predictions_scaled).reshape(-1, 1)
    predictions_original = scaler_Y.inverse_transform(predictions_scaled_reshaped)
    # reshape predictions_original to 1d array
    predictions_original = predictions_original.reshape(-1)

    # Calculating performance metrics
    r2_performance = r2_score(data[y_column], predictions_scaled)
    rmse_performance = np.sqrt(mean_squared_error(data[y_column], predictions_scaled)) * -1
    mae_performance = mean_absolute_error(data[y_column], predictions_scaled) * -1
    pearson_r_test, _ = pearsonr(data[y_column], predictions_scaled)
    spearman_r_test, _ = spearmanr(data[y_column], predictions_scaled)

    # Calculating performance metrics in original scale
    r2_performance_original = r2_score(data_raw[y_column], predictions_original)
    rmse_performance_original = np.sqrt(mean_squared_error(data_raw[y_column], predictions_original)) * -1
    mae_performance_original = mean_absolute_error(data_raw[y_column], predictions_original) * -1
    pearson_r_test_original, _ = pearsonr(data_raw[y_column], predictions_original)
    spearman_r_test_original, _ = spearmanr(data_raw[y_column], predictions_original)

    # Printing performance metrics
    print(f"\nPrediction Scores: {model_name}")
    print("======================================")
    print(f"Average Test R^2: {r2_performance:.4f}")
    print(f"Average Test RMSE: {rmse_performance:.4f}")
    print(f"Average Test MAE: {mae_performance:.4f}")
    print(f"Average Test Pearson r: {pearson_r_test:.4f}")
    print(f"Average Test Spearman r: {spearman_r_test:.4f}")
    print("======================================\n")

    print(f"\nPrediction Scores in original scale: {model_name}")
    print("======================================")
    print(f"Average Test R^2: {r2_performance_original:.4f}")
    print(f"Average Test RMSE: {rmse_performance_original:.4f}")
    print(f"Average Test MAE: {mae_performance_original:.4f}")
    print(f"Average Test Pearson r: {pearson_r_test_original:.4f}")
    print(f"Average Test Spearman r: {spearman_r_test_original:.4f}")
    print("======================================\n")

    performance_metrics = {
        'Model': model_name,
        'Feature_Combination': feature_comb,
        'R2_Scaled': r2_performance,
        'RMSE_Scaled': rmse_performance,
        'MAE_Scaled': mae_performance,
        'Pearson_r_Scaled': pearson_r_test,
        'Spearman_r_Scaled': spearman_r_test,
        'R2_Original': r2_performance_original,
        'RMSE_Original': rmse_performance_original,
        'MAE_Original': mae_performance_original,
        'Pearson_r_Original': pearson_r_test_original,
        'Spearman_r_Original': spearman_r_test_original
    }

    # Scatter plot between predicted values and true targets
    sns.set(style="whitegrid")
    g = sns.jointplot(x=data[y_column], y=predictions_scaled, kind="reg", height=8, color='blue', joint_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.6, 'edgecolors': 'w', 'linewidth': 0.5}})
    # Annotating the plot with the metrics
    annotations = [
        f'Pearson r: {pearson_r_test:.4f}',
        f'Spearman r: {spearman_r_test:.4f}',
        # f'R^2: {r2_performance:.4f}'
    ]
    for i, annotation in enumerate(annotations, 1):
        g.ax_joint.annotate(annotation, xy=(0.05, 1 - 0.07 * i), xycoords='axes fraction', ha='left', va='top', fontsize=12, backgroundcolor='white')
    g.set_axis_labels('True Values', 'Predicted Values')
    plt.tight_layout()
    plt.savefig(case_results_path + feature_comb + '_prediction_scaled.png')
    plt.show()
    plt.close()

    # Scatter plot between predicted values and true targets in original scale
    sns.set(style="whitegrid")
    g = sns.jointplot(x=data_raw[y_column], y=predictions_original, kind="reg", height=8, color='blue', joint_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.6, 'edgecolors': 'w', 'linewidth': 0.5}})
    # Annotating the plot with the metrics
    annotations = [
        f'Pearson r: {pearson_r_test_original:.4f}',
        f'Spearman r: {spearman_r_test_original:.4f}',
        # f'R^2: {r2_performance_original:.4f}'
    ]
    for i, annotation in enumerate(annotations, 1):
        g.ax_joint.annotate(annotation, xy=(0.05, 1 - 0.07 * i), xycoords='axes fraction', ha='left', va='top', fontsize=12, backgroundcolor='white')
    g.set_axis_labels('True Values', 'Predicted Values')
    plt.tight_layout()
    plt.savefig(case_results_path + feature_comb + '_prediction_original.png')
    plt.show()
    plt.close()

    return pd.DataFrame([performance_metrics])
