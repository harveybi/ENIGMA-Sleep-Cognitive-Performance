import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind

# %%
# reload utils
import importlib
importlib.reload(utils)

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'

# %%
# df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed.csv')
df_Liege_cleaned = pd.read_csv(data_save_path + 'Liege_dataset_renamed_cleaned.csv')
df_Liege_renamed = pd.read_csv(data_save_path + 'Liege_dataset_renamed.csv')
df_Liege_raw = pd.read_excel('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/Liege/Liege_ENIGMA_table_HB_Jul06.xlsx')

# %%
df_COF_Stroop = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/Liege/COF_Stroop_Reactiontimes.csv', sep=';')
df_COGNAP_Stroop = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/Liege/COGNAP_Stroop.csv', sep=';')

# %%
# for df_COF_Stroop add one column 'RT_diff' which is the 'RTI' - 'RTNE'
df_COF_Stroop['RT_diff'] = df_COF_Stroop['RTI'] - df_COF_Stroop['RTNE']

# %%
# devide 'RT_diff' by 1000
df_COF_Stroop['RT_diff'] = df_COF_Stroop['RT_diff'] / 10

# %%
# plot the distribution of 'RT_diff' in df_COF_Stroop, and 'Stroop_Interf_Denom' in df_COGNAP_Stroop
plt.figure(figsize=(10, 6))
sns.histplot(df_COF_Stroop['RT_diff'], color='blue', label='COF_Stroop', kde=True)
sns.histplot(df_COGNAP_Stroop['Stroop_Interf_Denom'], color='red', label='COGNAP_Stroop', kde=True)
plt.title('Distribution of RT_diff in COF_Stroop and Stroop_Interf_Denom in COGNAP_Stroop')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# %%
# plot 'Stroop_Test' in df_Liege_renamed
plt.figure(figsize=(10, 6))
sns.histplot(df_Liege_renamed['Stroop_Test'], color='blue', kde=True)
plt.title('Distribution of Stroop_Test in df_Liege_renamed')
plt.tight_layout()
plt.show()
plt.close()

# %%
def liege_sub_feature_check(df, feature, dataframe_name=None):
    print('\n')
    if dataframe_name:
        print(dataframe_name)
    print('Feature:', feature)
    print('COFxxx:', df[df['Sub_ID'].str.contains('COF')][feature].describe())
    # print number of missing values
    print('COFxxx missing values:', df[df['Sub_ID'].str.contains('COF')][feature].isna().sum())
    print('COGNAPxxx:', df[df['Sub_ID'].str.contains('COGNAP')][feature].describe())
    # print number of missing values
    print('COGNAPxxx missing values:', df[df['Sub_ID'].str.contains('COGNAP')][feature].isna().sum())
    print('\n')


# discribe 'Stroop_Test' for Sub_ID format 'COFxxx' and 'COGNAPxxx'
liege_sub_feature_check(df_Liege_cleaned, 'Stroop_Test', 'df_Liege_cleaned')
liege_sub_feature_check(df_Liege_renamed, 'Stroop_Test', 'df_Liege_renamed')
liege_sub_feature_check(df_Liege_raw, 'Stroop_Test', 'df_Liege_raw')

# %%
def liege_sub_feature_distribution_plot(df, feature, dataframe_name=None):
    # remove missing values
    df = df.dropna(subset=[feature])
    # calculate t-test for 'Stroop_Test' between 'COFxxx' and 'COGNAPxxx'
    ttest_result = ttest_ind(df[df['Sub_ID'].str.contains('COF')][feature], df[df['Sub_ID'].str.contains('COGNAP')][feature])

    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['Sub_ID'].str.contains('COF')][feature], color='blue', label='COFxxx', kde=True)
    sns.histplot(df[df['Sub_ID'].str.contains('COGNAP')][feature], color='red', label='COGNAPxxx', kde=True)
    # add t-test score and p-value to the plot
    plt.text(0.5, 0.5, 't-test score: ' + str(ttest_result[0]) + '\np-value: ' + str(ttest_result[1]), fontsize=12, transform=plt.gca().transAxes)
    plt.title('Distribution of ' + feature + ' in ' + dataframe_name)
    plt.legend()
    plt.show()


# plot 'Stroop_Test' distribution for Sub_ID format 'COFxxx' and 'COGNAPxxx'
liege_sub_feature_distribution_plot(df_Liege_cleaned, 'Stroop_Test', 'df_Liege_cleaned')
liege_sub_feature_distribution_plot(df_Liege_renamed, 'Stroop_Test', 'df_Liege_renamed')
liege_sub_feature_distribution_plot(df_Liege_raw, 'Stroop_Test', 'df_Liege_raw')


# %%
# check 'Memory_Test'
liege_sub_feature_check(df_Liege_cleaned, 'Memory_Test', 'df_Liege_cleaned')
liege_sub_feature_check(df_Liege_renamed, 'Memory_Test', 'df_Liege_renamed')
liege_sub_feature_check(df_Liege_raw, 'Memory_Test', 'df_Liege_raw')

# plot 'Memory_Test' distribution for Sub_ID format 'COFxxx' and 'COGNAPxxx'
liege_sub_feature_distribution_plot(df_Liege_cleaned, 'Memory_Test', 'df_Liege_cleaned')
liege_sub_feature_distribution_plot(df_Liege_renamed, 'Memory_Test', 'df_Liege_renamed')
liege_sub_feature_distribution_plot(df_Liege_raw, 'Memory_Test', 'df_Liege_raw')

# %%
# plot 'Age_at_Scan' distribution for Sub_ID format 'COFxxx' and 'COGNAPxxx'
liege_sub_feature_distribution_plot(df_Liege_cleaned, 'Age_at_Scan', 'df_Liege_cleaned')
liege_sub_feature_distribution_plot(df_Liege_renamed, 'Age_at_Scan', 'df_Liege_renamed')
liege_sub_feature_distribution_plot(df_Liege_raw, 'Age_at_Scan', 'df_Liege_raw')
