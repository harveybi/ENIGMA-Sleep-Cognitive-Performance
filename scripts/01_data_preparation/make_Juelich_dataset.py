import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils

import pandas as pd

# %%
raw_data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/Juelich/'
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'

# %%
df_SleepLess = pd.read_excel(raw_data_save_path + 'Participants_SleepLess.xlsx')
df_SomnoSafe = pd.read_excel(raw_data_save_path + 'Participants_SomnoSafe.xlsx')

# concatenate the two datasets
df_Juelich = pd.concat([df_SleepLess, df_SomnoSafe], axis=0)

# %%
# make a list of the columns that all values are NaN
cols_to_drop = df_Juelich.columns[df_Juelich.isna().all()].tolist()
# drop the columns that all values are NaN
df_Juelich = df_Juelich.drop(columns=cols_to_drop)

# %%
df_Letter_N_Back_061018 = pd.read_excel(raw_data_save_path + 'Letter_N-Back_061018.xlsx')
df_SleepLess_3_back_spatial = pd.read_excel(raw_data_save_path + 'SleepLess_3-back_spatial.xlsx')
df_SomnoSafe_Schlafauswertungsmatrix_02072019_final = pd.read_excel(raw_data_save_path + 'SomnoSafe_Schlafauswertungsmatrix_02072019_final.xls')
df_Spatial_N_Back_06102018 = pd.read_excel(raw_data_save_path + 'Spatial_N-Back _06102018.xlsx')

# %%
"""
For df_SomnoSafe_Schlafauswertungsmatrix_02072019_final, create a new column 'Sub_ID'.
The value of 'Sub_ID' is the combination of 'Sub_ID' and 'Site_Name'.
Combination rule: 'sub-' + column 'Kennung''s value + column 'Exponum''s value (orginal value is 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, to 001, 002, 003, 004, 005, 006, 007, 008, 009, 010)
"""
df_SomnoSafe_Schlafauswertungsmatrix_02072019_final['Sub_ID'] = (
    'sub-' +
    df_SomnoSafe_Schlafauswertungsmatrix_02072019_final['Kennung'].astype(str) +
    df_SomnoSafe_Schlafauswertungsmatrix_02072019_final['Exponum'].apply(lambda x: str(x).zfill(3))
)

# Move the 'Sub_ID' column to the first position
columns = ['Sub_ID'] + [col for col in df_SomnoSafe_Schlafauswertungsmatrix_02072019_final.columns if col != 'Sub_ID']
df_SomnoSafe_Schlafauswertungsmatrix_02072019_final = df_SomnoSafe_Schlafauswertungsmatrix_02072019_final[columns]

# %%
"""
Create df_Juelich_SomnoSafe. Which include columns: 'Sub_ID', 'Site_Name', 'Acquisition_Date', 'Age_at_Scan', 'SEX', 'BMI',
       'Education', 'Hypertension', 'Depression_score', 'Alcohol', 'Smoking',
       'Marital_Status', 'PSG_Sleep_Dur', 'Self_Sleep_Dur', 'PSG_Sleep_Eff',
       'Self_Sleep_Eff', 'APOE4',
       'Memory_Test'
"""
# create a new dataframe with above column names
df_Juelich_SomnoSafe = pd.DataFrame(columns=['Sub_ID', 'Site_Name', 'Acquisition_Date', 'Age_at_Scan', 'SEX', 'BMI',
       'Education', 'Hypertension', 'Depression_score', 'Alcohol', 'Smoking', 'Marital_Status', 'PSG_Sleep_Dur',
       'Self_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Eff', 'APOE4', 'Memory_Test'])

# %%
df_Juelich_SomnoSafe['Sub_ID'] = df_SomnoSafe['Sub_ID']
df_Juelich_SomnoSafe['Site_Name'] = df_SomnoSafe['Site_Name']
df_Juelich_SomnoSafe['Acquisition_Date'] = df_SomnoSafe['Acquisition_Date']
df_Juelich_SomnoSafe['Age_at_Scan'] = df_SomnoSafe['Age_at_Scan_in_Years']
df_Juelich_SomnoSafe['SEX'] = df_SomnoSafe['SEX']
df_Juelich_SomnoSafe['BMI'] = df_SomnoSafe['BMI']
df_Juelich_SomnoSafe['Education'] = df_SomnoSafe['Education']
# df_Juelich_SomnoSafe['Depression_score'] = [None] * len(df_Juelich_SomnoSafe)
# df_Juelich_SomnoSafe['PSG_Sleep_Dur'] = [None] * len(df_Juelich_SomnoSafe)
# df_Juelich_SomnoSafe['Self_Sleep_Dur'] = [None] * len(df_Juelich_SomnoSafe)
# df_Juelich_SomnoSafe['PSG_Sleep_Eff'] = [None] * len(df_Juelich_SomnoSafe)
# df_Juelich_SomnoSafe['Self_Sleep_Eff'] = [None] * len(df_Juelich_SomnoSafe)
df_Juelich_SomnoSafe['PVT_reaction_speed'] = df_SomnoSafe['PVT_reaction_speed_baseline']
# df_Juelich_SomnoSafe['Memory_Test'] = [None] * len(df_Juelich_SomnoSafe)

# %%
df_merged = df_Juelich_SomnoSafe.merge(
    df_SomnoSafe_Schlafauswertungsmatrix_02072019_final[['Sub_ID', 'TST', 'SE']],
    on='Sub_ID',
    how='left'
)

# Map the values to the desired columns in df_Juelich_SomnoSafe
df_Juelich_SomnoSafe['PSG_Sleep_Dur'] = df_merged['TST']
df_Juelich_SomnoSafe['PSG_Sleep_Eff'] = df_merged['SE']

# %%
# save the df_Juelich_SomnoSafe to a csv file
# df_Juelich_SomnoSafe.to_csv(data_save_path + 'Juelich_SomnoSafe.csv')

# %%
# for 'Sub_ID' column, the value is like `sub-105006001`. I want to know how many not overlapped subjects, based on sub-105006.
# I will use the first 10 characters of 'Sub_ID' to get the unique subjects
unique_subjects = df_Juelich_SomnoSafe['Sub_ID'].apply(lambda x: x[:10]).unique()

# %%
# separate the df_Juelich_SomnoSafe to df_Juelich_SomnoSafe_1, df_Juelich_SomnoSafe_2, df_Juelich_SomnoSafe_3. Based on the index, 1 is row 0 to 35, 2 is row 36 to 71, 3 is row 72 to 107
df_Juelich_SomnoSafe_1 = df_Juelich_SomnoSafe.iloc[0:36]
df_Juelich_SomnoSafe_2 = df_Juelich_SomnoSafe.iloc[36:72]
df_Juelich_SomnoSafe_SD = df_Juelich_SomnoSafe.iloc[72:108]

# %%
unique_counts = df_Letter_N_Back_061018.groupby('condition')['Kennung'].nunique()
print(unique_counts)

# %%
# make sure the 'Kennung' column in df_Letter_N_Back_061018 and df_Spatial_N_Back_06102018 are string type
df_Letter_N_Back_061018['Kennung'] = df_Letter_N_Back_061018['Kennung'].astype(str)
df_Spatial_N_Back_06102018['Kennung'] = df_Spatial_N_Back_06102018['Kennung'].astype(str)

# %%
# Get unique subjects for condition = 0
letter_subjects_condition_0 = df_Letter_N_Back_061018.loc[df_Letter_N_Back_061018['condition'] == 0, 'Kennung'].unique()
spatial_subjects_condition_0 = df_Spatial_N_Back_06102018.loc[df_Spatial_N_Back_06102018['condition'] == 0, 'Kennung'].unique()

# Get unique subjects for condition = 1
letter_subjects_condition_1 = df_Letter_N_Back_061018.loc[df_Letter_N_Back_061018['condition'] == 1, 'Kennung'].unique()
spatial_subjects_condition_1 = df_Spatial_N_Back_06102018.loc[df_Spatial_N_Back_06102018['condition'] == 1, 'Kennung'].unique()

print("Subjects with condition = 0:")
print(letter_subjects_condition_0)
print(spatial_subjects_condition_0)

print("Subjects with condition = 1:")
print(letter_subjects_condition_1)
print(spatial_subjects_condition_1)

# %%
# if letter_subjects_condition_0 and spatial_subjects_condition_0 are same, make a new list subjects_condition_0
subjects_condition_0 = list(set(letter_subjects_condition_0) & set(spatial_subjects_condition_0))

# if letter_subjects_condition_1 and spatial_subjects_condition_1 are same, make a new list subjects_condition_1
subjects_condition_1 = list(set(letter_subjects_condition_1) & set(spatial_subjects_condition_1))

# %%
# for df_Juelich_SomnoSafe_1, df_Juelich_SomnoSafe_2, df_Juelich_SomnoSafe_SD's `Sub_ID` column, the value is like `sub-105006001`, now transfer to like sub-105006.
df_Juelich_SomnoSafe_1['Sub_ID'] = df_Juelich_SomnoSafe_1['Sub_ID'].apply(lambda x: x[:10])
df_Juelich_SomnoSafe_2['Sub_ID'] = df_Juelich_SomnoSafe_2['Sub_ID'].apply(lambda x: x[:10])
df_Juelich_SomnoSafe_SD['Sub_ID'] = df_Juelich_SomnoSafe_SD['Sub_ID'].apply(lambda x: x[:10])

# %%
# Add 'sub-' prefix to subjects in subjects_condition_0 and subjects_condition_1
subjects_condition_0 = ['sub-' + str(s) for s in subjects_condition_0]
subjects_condition_1 = ['sub-' + str(s) for s in subjects_condition_1]

# Convert lists to sets for faster lookup
subjects_condition_0_set = set(subjects_condition_0)
subjects_condition_1_set = set(subjects_condition_1)

def assign_condition(sub_id):
    if sub_id in subjects_condition_0_set:
        return 0
    elif sub_id in subjects_condition_1_set:
        return 1
    else:
        return None  # or another default value if the subject isn't found

# Apply to each DataFrame
df_Juelich_SomnoSafe_1['condition'] = df_Juelich_SomnoSafe_1['Sub_ID'].apply(assign_condition)
df_Juelich_SomnoSafe_2['condition'] = df_Juelich_SomnoSafe_2['Sub_ID'].apply(assign_condition)
df_Juelich_SomnoSafe_SD['condition'] = df_Juelich_SomnoSafe_SD['Sub_ID'].apply(assign_condition)

# %%
# A small helper function to reorder columns
def move_condition_next_to_sub_id(df):
    cols = df.columns.tolist()
    # Remove 'condition' from the list (it should already exist)
    cols.remove('condition')
    # Find the index of 'Sub_ID'
    sub_id_idx = cols.index('Sub_ID')
    # Insert 'condition' right after 'Sub_ID'
    cols.insert(sub_id_idx + 1, 'condition')
    return df[cols]

# Apply to each DataFrame
df_Juelich_SomnoSafe_1 = move_condition_next_to_sub_id(df_Juelich_SomnoSafe_1)
df_Juelich_SomnoSafe_2 = move_condition_next_to_sub_id(df_Juelich_SomnoSafe_2)
df_Juelich_SomnoSafe_SD = move_condition_next_to_sub_id(df_Juelich_SomnoSafe_SD)

# %%
# Calculate Memory test score
# Step 1: Extract Kennung from Sub_ID (remove 'sub-' prefix)
df_Juelich_SomnoSafe_1['Kennung'] = df_Juelich_SomnoSafe_1['Sub_ID'].str.replace('sub-', '', regex=False)
df_Juelich_SomnoSafe_2['Kennung'] = df_Juelich_SomnoSafe_2['Sub_ID'].str.replace('sub-', '', regex=False)
df_Juelich_SomnoSafe_SD['Kennung'] = df_Juelich_SomnoSafe_SD['Sub_ID'].str.replace('sub-', '', regex=False)

# Step 2: Reformat Acquisition_Date from 'YYYY-MM-DD' to 'YYYYMMDD'
df_Juelich_SomnoSafe_1['date_join'] = df_Juelich_SomnoSafe_1['Acquisition_Date'].dt.strftime('%Y%m%d')
df_Juelich_SomnoSafe_2['date_join'] = df_Juelich_SomnoSafe_2['Acquisition_Date'].dt.strftime('%Y%m%d')
df_Juelich_SomnoSafe_SD['date_join'] = df_Juelich_SomnoSafe_SD['Acquisition_Date'].dt.strftime('%Y%m%d')

# %%
# print the type of 'Kennung' 'date' column in df_Letter_N_Back_061018
print(df_Letter_N_Back_061018['Kennung'].dtype)
print(df_Letter_N_Back_061018['date'].dtype)

# print the type of 'Kennung' 'date_join' column in df_Juelich_SomnoSafe_1
print(df_Juelich_SomnoSafe_1['Kennung'].dtype)
print(df_Juelich_SomnoSafe_1['date_join'].dtype)

# unify the type of 'Kennung' and 'date' columns as string
df_Letter_N_Back_061018['Kennung'] = df_Letter_N_Back_061018['Kennung'].astype(str)
df_Letter_N_Back_061018['date'] = df_Letter_N_Back_061018['date'].astype(str)

df_Spatial_N_Back_06102018['Kennung'] = df_Spatial_N_Back_06102018['Kennung'].astype(str)
df_Spatial_N_Back_06102018['date'] = df_Spatial_N_Back_06102018['date'].astype(str)

# unify the type of 'Kennung' and 'date_join' columns as string
df_Juelich_SomnoSafe_1['Kennung'] = df_Juelich_SomnoSafe_1['Kennung'].astype(str)
df_Juelich_SomnoSafe_1['date_join'] = df_Juelich_SomnoSafe_1['date_join'].astype(str)

# %%
# Create df_Letter_N_Back_061018_filtered_1
df_Letter_N_Back_061018_filtered_1 = df_Letter_N_Back_061018.merge(
    df_Juelich_SomnoSafe_1[['Kennung', 'date_join']],  # only keep columns needed for matching
    left_on=['Kennung', 'date'],                       # matching keys in df_Letter_N_Back_061018
    right_on=['Kennung', 'date_join'],                 # matching keys in df_Juelich_SomnoSafe_1
    how='inner'                                        # only keep matching rows
)
# Drop the extra 'date_join' column from the merged result
df_Letter_N_Back_061018_filtered_1.drop('date_join', axis=1, inplace=True)

# Repeat the same logic for df_Juelich_SomnoSafe_2 and df_Juelich_SomnoSafe_SD
df_Letter_N_Back_061018_filtered_2 = df_Letter_N_Back_061018.merge(
    df_Juelich_SomnoSafe_2[['Kennung', 'date_join']],
    left_on=['Kennung', 'date'],
    right_on=['Kennung', 'date_join'],
    how='inner'
)
df_Letter_N_Back_061018_filtered_2.drop('date_join', axis=1, inplace=True)

df_Letter_N_Back_061018_filtered_SD = df_Letter_N_Back_061018.merge(
    df_Juelich_SomnoSafe_SD[['Kennung', 'date_join']],
    left_on=['Kennung', 'date'],
    right_on=['Kennung', 'date_join'],
    how='inner'
)
df_Letter_N_Back_061018_filtered_SD.drop('date_join', axis=1, inplace=True)

# %%
# create df_Spatial_N_Back_06102018_filtered_1, df_Spatial_N_Back_06102018_filtered_2, df_Spatial_N_Back_06102018_filtered_SD
df_Spatial_N_Back_06102018_filtered_1 = df_Spatial_N_Back_06102018.merge(
    df_Juelich_SomnoSafe_1[['Kennung', 'date_join']],
    left_on=['Kennung', 'date'],
    right_on=['Kennung', 'date_join'],
    how='inner'
)
df_Spatial_N_Back_06102018_filtered_1.drop('date_join', axis=1, inplace=True)

df_Spatial_N_Back_06102018_filtered_2 = df_Spatial_N_Back_06102018.merge(
    df_Juelich_SomnoSafe_2[['Kennung', 'date_join']],
    left_on=['Kennung', 'date'],
    right_on=['Kennung', 'date_join'],
    how='inner'
)
df_Spatial_N_Back_06102018_filtered_2.drop('date_join', axis=1, inplace=True)

df_Spatial_N_Back_06102018_filtered_SD = df_Spatial_N_Back_06102018.merge(
    df_Juelich_SomnoSafe_SD[['Kennung', 'date_join']],
    left_on=['Kennung', 'date'],
    right_on=['Kennung', 'date_join'],
    how='inner'
)
df_Spatial_N_Back_06102018_filtered_SD.drop('date_join', axis=1, inplace=True)

# %%
"""
df_Juelich_SomnoSafe_1, df_Letter_N_Back_061018_filtered_1
df_Juelich_SomnoSafe_2, df_Letter_N_Back_061018_filtered_2
df_Juelich_SomnoSafe_SD, df_Letter_N_Back_061018_filtered_SD
"""
# for each subject in df_Juelich_SomnoSafe_1, based on 'Kennung', find the corresponding rows of this subject in df_Letter_N_Back_061018_filtered_1.
# each subject will have at least 15 rows. I want the average value of last 12 rows of 'hit rate = number of correct match trials /number of match trials', as 'Letter_HR' in df_Juelich_SomnoSafe_1;
# the average value of last 12 rows of 'Rtmean', as 'Letter_Rtmean' in df_Juelich_SomnoSafe_1; the average value of last 12 rows of 'A\ ', as 'Letter_Sensitivity' in df_Juelich_SomnoSafe_1.
def letter_last_12_stats(group):
    # Select the last 12 rows for the subject
    last_12 = group.tail(12)

    # Compute averages
    Letter_HR = last_12['hit rate = number of correct match trials /number of match trials'].mean()
    Letter_Rtmean = last_12['Rtmean'].mean()
    Letter_Sensitivity = last_12['A\' '].mean()  # Adjust the column name if needed

    return pd.Series({
        'Letter_HR': Letter_HR,
        'Letter_Rtmean': Letter_Rtmean,
        'Letter_Sensitivity': Letter_Sensitivity
    })


letter_stats_1 = df_Letter_N_Back_061018_filtered_1.groupby('Kennung').apply(last_12_stats).reset_index()
df_Juelich_SomnoSafe_1 = df_Juelich_SomnoSafe_1.merge(letter_stats_1, on='Kennung', how='left')

letter_stats_2 = df_Letter_N_Back_061018_filtered_2.groupby('Kennung').apply(last_12_stats).reset_index()
df_Juelich_SomnoSafe_2 = df_Juelich_SomnoSafe_2.merge(letter_stats_2, on='Kennung', how='left')

letter_stats_SD = df_Letter_N_Back_061018_filtered_SD.groupby('Kennung').apply(last_12_stats).reset_index()
df_Juelich_SomnoSafe_SD = df_Juelich_SomnoSafe_SD.merge(letter_stats_SD, on='Kennung', how='left')

# %%
"""
df_Juelich_SomnoSafe_1, df_Spatial_N_Back_06102018_filtered_1
df_Juelich_SomnoSafe_2, df_Spatial_N_Back_06102018_filtered_2
df_Juelich_SomnoSafe_SD, df_Spatial_N_Back_06102018_filtered_SD
"""
def spatial_last_12_stats(group):
    # Select the last 12 rows for the subject
    last_12 = group.tail(12)

    # Compute averages
    Spatial_HR = last_12['hit rate = number of correct match trials /number of match trials'].mean()
    Spatial_Rtmean = last_12['Rtmean'].mean()
    Spatial_Sensitivity = last_12['A\' '].mean()  # Adjust the column name if needed

    return pd.Series({
        'Spatial_HR': Spatial_HR,
        'Spatial_Rtmean': Spatial_Rtmean,
        'Spatial_Sensitivity': Spatial_Sensitivity
    })

spatial_stats_1 = df_Spatial_N_Back_06102018_filtered_1.groupby('Kennung').apply(spatial_last_12_stats).reset_index()
df_Juelich_SomnoSafe_1 = df_Juelich_SomnoSafe_1.merge(spatial_stats_1, on='Kennung', how='left')

spatial_stats_2 = df_Spatial_N_Back_06102018_filtered_2.groupby('Kennung').apply(spatial_last_12_stats).reset_index()
df_Juelich_SomnoSafe_2 = df_Juelich_SomnoSafe_2.merge(spatial_stats_2, on='Kennung', how='left')

spatial_stats_SD = df_Spatial_N_Back_06102018_filtered_SD.groupby('Kennung').apply(spatial_last_12_stats).reset_index()
df_Juelich_SomnoSafe_SD = df_Juelich_SomnoSafe_SD.merge(spatial_stats_SD, on='Kennung', how='left')

# %%
# save the df_Juelich_SomnoSafe_1, df_Juelich_SomnoSafe_2, df_Juelich_SomnoSafe_SD to csv files
df_Juelich_SomnoSafe_1.to_csv(data_save_path + 'Juelich_SomnoSafe_1.csv')
df_Juelich_SomnoSafe_2.to_csv(data_save_path + 'Juelich_SomnoSafe_2.csv')
df_Juelich_SomnoSafe_SD.to_csv(data_save_path + 'Juelich_SomnoSafe_SD.csv')

# %%
# load this three csv files
df_Juelich_SomnoSafe_1 = pd.read_csv(data_save_path + 'Juelich_SomnoSafe_1.csv', index_col=0)
df_Juelich_SomnoSafe_2 = pd.read_csv(data_save_path + 'Juelich_SomnoSafe_2.csv', index_col=0)
df_Juelich_SomnoSafe_SD = pd.read_csv(data_save_path + 'Juelich_SomnoSafe_SD.csv', index_col=0)

# %%
# for these three dataframe, drop 'Letter_HR_y', 'Letter_Rtmean_y', 'Letter_Sensitivity_y' columns.
df_Juelich_SomnoSafe_1.drop(['Letter_HR_y', 'Letter_Rtmean_y', 'Letter_Sensitivity_y'], axis=1, inplace=True)
df_Juelich_SomnoSafe_2.drop(['Letter_HR_y', 'Letter_Rtmean_y', 'Letter_Sensitivity_y'], axis=1, inplace=True)
df_Juelich_SomnoSafe_SD.drop(['Letter_HR_y', 'Letter_Rtmean_y', 'Letter_Sensitivity_y'], axis=1, inplace=True)

# rename 'Letter_HR_x', 'Letter_Rtmean_x', 'Letter_Sensitivity_x' columns to 'Letter_HR', 'Letter_Rtmean', 'Letter_Sensitivity'
df_Juelich_SomnoSafe_1.rename(columns={'Letter_HR_x': 'Letter_HR', 'Letter_Rtmean_x': 'Letter_Rtmean', 'Letter_Sensitivity_x': 'Letter_Sensitivity'}, inplace=True)
df_Juelich_SomnoSafe_2.rename(columns={'Letter_HR_x': 'Letter_HR', 'Letter_Rtmean_x': 'Letter_Rtmean', 'Letter_Sensitivity_x': 'Letter_Sensitivity'}, inplace=True)
df_Juelich_SomnoSafe_SD.rename(columns={'Letter_HR_x': 'Letter_HR', 'Letter_Rtmean_x': 'Letter_Rtmean', 'Letter_Sensitivity_x': 'Letter_Sensitivity'}, inplace=True)

# %%
# save back
df_Juelich_SomnoSafe_1.to_csv(data_save_path + 'Juelich_SomnoSafe_1.csv')
df_Juelich_SomnoSafe_2.to_csv(data_save_path + 'Juelich_SomnoSafe_2.csv')
df_Juelich_SomnoSafe_SD.to_csv(data_save_path + 'Juelich_SomnoSafe_SD.csv')
