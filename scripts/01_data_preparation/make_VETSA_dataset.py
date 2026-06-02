import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# %%
raw_data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/San Diago/ENIGMA_Sleep_Masoud_2024_10_14/files/'
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'

# %%
df_demo_VETSA = pd.read_csv(raw_data_save_path + 'ESR_01_ENIGMA_Sleep_nonMRIdata_revised.csv')
df_area_DK_VETSA = pd.read_csv(raw_data_save_path + 'ESR_02_UCSD_area_DK.csv')
df_area_Schaefer_VETSA = pd.read_csv(raw_data_save_path + 'ESR_03_UCSD_area_Schaefer.csv')
df_subcor_VETSA = pd.read_csv(raw_data_save_path + 'ESR_04_UCSD_subcortical_volume.csv')
df_thickness_DK_VETSA = pd.read_csv(raw_data_save_path + 'ESR_05_UCSD_thickness_DK.csv')
df_thickness_Schaefer_VETSA = pd.read_csv(raw_data_save_path + 'ESR_06_UCSD_thickness_Schaefer.csv')

# %%
# for all dataframes, rename 'CID' to 'Sub_ID', then sort by 'Sub_ID'
df_demo_VETSA.rename(columns={'CID': 'Sub_ID'}, inplace=True)
df_demo_VETSA.sort_values(by='Sub_ID', inplace=True)
df_area_DK_VETSA.rename(columns={'CID': 'Sub_ID'}, inplace=True)
df_area_DK_VETSA.sort_values(by='Sub_ID', inplace=True)
df_area_Schaefer_VETSA.rename(columns={'CID': 'Sub_ID'}, inplace=True)
df_area_Schaefer_VETSA.sort_values(by='Sub_ID', inplace=True)
df_subcor_VETSA.rename(columns={'CID': 'Sub_ID'}, inplace=True)
df_subcor_VETSA.sort_values(by='Sub_ID', inplace=True)
df_thickness_DK_VETSA.rename(columns={'CID': 'Sub_ID'}, inplace=True)
df_thickness_DK_VETSA.sort_values(by='Sub_ID', inplace=True)
df_thickness_Schaefer_VETSA.rename(columns={'CID': 'Sub_ID'}, inplace=True)
df_thickness_Schaefer_VETSA.sort_values(by='Sub_ID', inplace=True)

# %%
"""
Change column names for df_demo_VETSA:
'cesdtot_V3' to 'Depression_score'
'BMI_V3' to 'BMI'
'Age_V3' to 'Age_at_Scan'
'apoe2024' to 'APOE4'
'HRSSLEEP_V3' to 'Self_Sleep_Dur'
'sleepeff_V3' to 'Self_Sleep_Eff'
'DSFRAW_V3' to 'Digit Span Forward Raw'
'DSBRAW_V3' to 'Digit Span Backward Raw'
'STRWRAW_V3' to 'Stroop Raw Word Score'
'STRCRAW_V3' to 'Stroop Raw Color Score'
'STRCWRAW_V3' to 'Stroop Raw Color-Word Score'
'DSTOT_V3p' to 'Digit Span Total Trials Passed'
'LNTOT_V3p' to 'Letter-Number Sequencing Total Score'
'STRIT_V3p' to 'Stroop Interference Norm-Based T-Score'
"""
df_demo_VETSA.rename(columns={'cesdtot_V3': 'Depression_score'}, inplace=True)
df_demo_VETSA.rename(columns={'BMI_V3': 'BMI'}, inplace=True)
df_demo_VETSA.rename(columns={'AGE_V3': 'Age_at_Scan'}, inplace=True)
df_demo_VETSA.rename(columns={'apoe2024': 'APOE4'}, inplace=True)
df_demo_VETSA.rename(columns={'HRSSLEEP_V3': 'Self_Sleep_Dur'}, inplace=True)
df_demo_VETSA.rename(columns={'sleepeff_V3': 'Self_Sleep_Eff'}, inplace=True)
df_demo_VETSA.rename(columns={'DSFRAW_V3': 'Digit Span Forward Raw'}, inplace=True)
df_demo_VETSA.rename(columns={'DSBRAW_V3': 'Digit Span Backward Raw'}, inplace=True)
df_demo_VETSA.rename(columns={'STRWRAW_V3': 'Stroop Raw Word Score'}, inplace=True)
df_demo_VETSA.rename(columns={'STRCRAW_V3': 'Stroop Raw Color Score'}, inplace=True)
df_demo_VETSA.rename(columns={'STRCWRAW_V3': 'Stroop Raw Color-Word Score'}, inplace=True)
df_demo_VETSA.rename(columns={'DSTOT_V3p': 'Digit Span Total Trials Passed'}, inplace=True)
df_demo_VETSA.rename(columns={'LNTOT_V3p': 'Letter-Number Sequencing Total Score'}, inplace=True)
df_demo_VETSA.rename(columns={'STRIT_V3p': 'Stroop Interference Norm-Based T-Score'}, inplace=True)

# %%
# for df_demo_VETSA, df_thickness_DK_VETSA, df_thickness_Schaefer_VETSA, df_area_DK_VETSA, df_area_Schaefer_VETSA, df_subcor_VETSA, check whether 'Sub_ID' column has unique values
# Create a set of Sub_IDs for each DataFrame
sub_ids = {
    "df_demo_VETSA": set(df_demo_VETSA['Sub_ID']),
    "df_thickness_DK_VETSA": set(df_thickness_DK_VETSA['Sub_ID']),
    "df_thickness_Schaefer_VETSA": set(df_thickness_Schaefer_VETSA['Sub_ID']),
    "df_area_DK_VETSA": set(df_area_DK_VETSA['Sub_ID']),
    "df_area_Schaefer_VETSA": set(df_area_Schaefer_VETSA['Sub_ID']),
    "df_subcor_VETSA": set(df_subcor_VETSA['Sub_ID']),
}

# Find common Sub_IDs across all DataFrames
common_sub_ids = set.intersection(*sub_ids.values())
print(f"Common Sub_IDs across all DataFrames: {len(common_sub_ids)}")

# Find unique Sub_IDs for each DataFrame
unique_sub_ids = {key: value - common_sub_ids for key, value in sub_ids.items()}
for df_name, unique_ids in unique_sub_ids.items():
    print(f"Unique Sub_IDs in {df_name}: {len(unique_ids)}")

# Check which DataFrames have more or less subjects
sub_id_counts = {df_name: len(ids) for df_name, ids in sub_ids.items()}
print("Number of Sub_IDs in each DataFrame:")
for df_name, count in sub_id_counts.items():
    print(f"{df_name}: {count}")

# Identify differences between DataFrames
for df1, ids1 in sub_ids.items():
    for df2, ids2 in sub_ids.items():
        if df1 != df2:
            extra_in_df1 = ids1 - ids2
            extra_in_df2 = ids2 - ids1
            print(f"Subjects in {df1} but not in {df2}: {len(extra_in_df1)}")
            print(f"Subjects in {df2} but not in {df1}: {len(extra_in_df2)}")

# Identify and display the specific Sub_ID values
for df1, ids1 in sub_ids.items():
    for df2, ids2 in sub_ids.items():
        if df1 != df2:
            extra_in_df1 = ids1 - ids2  # Sub_IDs in df1 but not in df2
            extra_in_df2 = ids2 - ids1  # Sub_IDs in df2 but not in df1
            if extra_in_df1:
                print(f"Sub_IDs in {df1} but not in {df2} ({len(extra_in_df1)}): {sorted(extra_in_df1)}")
            if extra_in_df2:
                print(f"Sub_IDs in {df2} but not in {df1} ({len(extra_in_df2)}): {sorted(extra_in_df2)}")

# %%
# remove subject 51011 from df_demo_VETSA
df_demo_VETSA = df_demo_VETSA[df_demo_VETSA['Sub_ID'] != 51011]

# %%
"""
Rename of imaging tables
"""
label_subcortical = ['Sub_ID', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
                     'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
                     '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
                     'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel', 'Right-Lateral-Ventricle',
                     'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
                     'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
                     'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel', '5th-Ventricle',
                     'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
                     'EstimatedTotalIntraCranialVol']
df_subcor_VETSA = df_subcor_VETSA[label_subcortical]

# %%
df_subcor_VETSA = df_subcor_VETSA.rename(columns={'Left-Thalamus-Proper': 'Left-Thalamus',
                                                  'Right-Thalamus-Proper': 'Right-Thalamus',
                                                  'subject_ID': 'Sub_ID'})

# %%
# rename cortical thickness
# Schaefer
filtered_columns = [col for col in df_thickness_Schaefer_VETSA.columns if
                    col not in ['Sub_ID', 'BrainSegVolNotVent', 'eTIV']]
renamed_schaefer_ct_df_columns = [col.replace('lh_7Networks_', '').replace('rh_7Networks_', '') for col in
                                  filtered_columns]
# DK
renamed_dk_ct_df_columns = [col for col in df_thickness_DK_VETSA.columns if
                            col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV']]

# rename surface area
# Schaefer
filtered_columns = [col for col in df_area_Schaefer_VETSA.columns if
                    col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV', 'lh_WhiteSurfArea_area',
                                'rh_WhiteSurfArea_area']]
renamed_schaefer_sa_df_columns = [col.replace('lh_7Networks_', '').replace('rh_7Networks_', '') for col in
                                  filtered_columns]
# DK
renamed_dk_sa_df_columns = [col for col in df_area_DK_VETSA.columns if
                            col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV']]

# %%
# apply the renaming and filtering
# Rename cortical thickness columns for Schaefer
filtered_ct_columns_schaefer = [col for col in df_thickness_Schaefer_VETSA.columns if
                                col not in ['Sub_ID', 'BrainSegVolNotVent', 'eTIV']]
renamed_ct_columns_schaefer = [col.replace('lh_7Networks_', '').replace('rh_7Networks_', '') for col in
                                filtered_ct_columns_schaefer]
df_thickness_Schaefer_VETSA.rename(columns=dict(zip(filtered_ct_columns_schaefer, renamed_ct_columns_schaefer)), inplace=True)
df_thickness_Schaefer_VETSA = df_thickness_Schaefer_VETSA[['Sub_ID'] + renamed_ct_columns_schaefer]

# Rename cortical thickness columns for DK
filtered_ct_columns_dk = [col for col in df_thickness_DK_VETSA.columns if
                          col not in ['Sub_ID', 'BrainSegVolNotVent', 'eTIV']]
# No specific replacement given for DK; keeping original names
df_thickness_DK_VETSA = df_thickness_DK_VETSA[['Sub_ID'] + filtered_ct_columns_dk]

# Rename surface area columns for Schaefer
filtered_sa_columns_schaefer = [col for col in df_area_Schaefer_VETSA.columns if
                                 col not in ['Sub_ID', 'BrainSegVolNotVent', 'eTIV', 'lh_WhiteSurfArea_area', 'rh_WhiteSurfArea_area']]
renamed_sa_columns_schaefer = [col.replace('lh_7Networks_', '').replace('rh_7Networks_', '') for col in
                                filtered_sa_columns_schaefer]
df_area_Schaefer_VETSA.rename(columns=dict(zip(filtered_sa_columns_schaefer, renamed_sa_columns_schaefer)), inplace=True)
df_area_Schaefer_VETSA = df_area_Schaefer_VETSA[['Sub_ID'] + renamed_sa_columns_schaefer]

# Rename surface area columns for DK
filtered_sa_columns_dk = [col for col in df_area_DK_VETSA.columns if
                          col not in ['Sub_ID', 'BrainSegVolNotVent', 'eTIV']]
# No specific replacement given for DK; keeping original names
df_area_DK_VETSA = df_area_DK_VETSA[['Sub_ID'] + filtered_sa_columns_dk]

# %%
# concatenate all dataframes by 'Sub_ID' column. Order of dataframes is df_demo_VETSA, df_thickness_DK_VETSA, df_thickness_Schaefer_VETSA, df_area_DK_VETSA, df_area_Schaefer_VETSA, df_subcor_VETSA
# Sequentially merge all DataFrames by 'Sub_ID'
df_combined = df_demo_VETSA.copy()  # Start with df_demo_VETSA as the base

# Merge with each subsequent DataFrame
df_combined = df_combined.merge(df_thickness_DK_VETSA, on='Sub_ID', how='outer')
df_combined = df_combined.merge(df_thickness_Schaefer_VETSA, on='Sub_ID', how='outer')
df_combined = df_combined.merge(df_area_DK_VETSA, on='Sub_ID', how='outer')
df_combined = df_combined.merge(df_area_Schaefer_VETSA, on='Sub_ID', how='outer')
df_combined = df_combined.merge(df_subcor_VETSA, on='Sub_ID', how='outer')

# %%
# Transform the APOE4 column: 1 for carrier (if '4' is present), 2 for non-carrier
# Safely transform the APOE4 column, handling NaN values
def transform_apoe4(value):
    try:
        # Check if value is NaN
        if pd.isna(value):
            return None  # Or any other representation for missing values, e.g., 'Unknown'
        # Check if '4' is present in the string
        return '1' if '4' in str(value) else '2'
    except Exception as e:
        print(f"Error processing value: {value}, Error: {e}")
        return None  # Handle unexpected cases gracefully

# Apply the function
df_combined['APOE4'] = df_combined['APOE4'].apply(transform_apoe4)

# %%
# for df_combined, make a new column 'Stroop_Test', which is same as 'Stroop Interference Norm-Based T-Score'
df_combined['Stroop_Test'] = df_combined['Stroop Interference Norm-Based T-Score']

# %%
# Get the list of columns
columns = df_combined.columns.tolist()

# Remove 'Stroop_Test' from the columns list
columns.remove('Stroop_Test')

# Find the index of 'Stroop Interference Norm-Based T-Score'
index = columns.index('Stroop Interference Norm-Based T-Score')

# Insert 'Stroop_Test' immediately after 'Stroop Interference Norm-Based T-Score'
columns.insert(index + 1, 'Stroop_Test')

# Reorder the DataFrame
df_combined = df_combined[columns]

# %%
# description of 'Digit Span Forward Raw', 'Digit Span Backward Raw'
df_combined['Digit Span Forward Raw'] = df_combined['Digit Span Forward Raw'].astype(float)
df_combined['Digit Span Backward Raw'] = df_combined['Digit Span Backward Raw'].astype(float)
print(df_combined['Digit Span Forward Raw'].describe())
print(df_combined['Digit Span Backward Raw'].describe())

# %%
# add a column 'Memory_Test_Digit'. The value of this column is the sum of 'Digit Span Forward Raw' and 'Digit Span Backward Raw' then divided by 30
df_combined['Memory_Test_Digit'] = (df_combined['Digit Span Forward Raw'] + df_combined['Digit Span Backward Raw']) / 30

print(df_combined['Memory_Test_Digit'].describe())

# %%
# Get the list of columns
columns = df_combined.columns.tolist()

# Remove 'Stroop_Test' from the columns list
columns.remove('Memory_Test_Digit')

# Find the index of 'Stroop Interference Norm-Based T-Score'
index = columns.index('Stroop_Test')

# Insert 'Stroop_Test' immediately after 'Stroop Interference Norm-Based T-Score'
columns.insert(index + 1, 'Memory_Test_Digit')

# Reorder the DataFrame
df_combined = df_combined[columns]

# %%
print(df_combined['Letter-Number Sequencing Total Score'].describe())

# %%
df_combined['Memory_Test_Letter'] = df_combined['Letter-Number Sequencing Total Score'] / 21

# %%
# Get the list of columns
columns = df_combined.columns.tolist()

# Remove 'Stroop_Test' from the columns list
columns.remove('Memory_Test_Letter')

# Find the index of 'Stroop Interference Norm-Based T-Score'
index = columns.index('Memory_Test_Digit')

# Insert 'Stroop_Test' immediately after 'Stroop Interference Norm-Based T-Score'
columns.insert(index + 1, 'Memory_Test_Letter')

# Reorder the DataFrame
df_combined = df_combined[columns]

# %%
# add 'SEX' column to the dataframe, all values are 1
df_combined['SEX'] = '1'

# %%
# save the df_combined to a csv file
# df_combined.to_csv(data_save_path + 'VETSA_dataset_renamed.csv')

# %%
# load the saved csv file
df_combined = pd.read_csv(data_save_path + 'VETSA_dataset_renamed.csv')

# %%
# print range of Stroop
print(f"Range of Stroop_Test: {df_combined['Stroop_Test'].min()} - {df_combined['Stroop_Test'].max()}")

# plot histogram of Stroop_Test
plt.figure(figsize=(8, 6))
sns.histplot(df_combined['Stroop_Test'].dropna(), bins=30, kde=True)
plt.title('Histogram of Stroop_Test')
plt.xlabel('Stroop_Test Score')
plt.ylabel('Frequency')
plt.show()

min_val = df_combined['Stroop_Test'].min()  # 20.0
max_val = df_combined['Stroop_Test'].max()  # 65.250882948

# Transform the Stroop_Test so that smaller values correspond to better performance
df_combined['Stroop_Test'] = (max_val + min_val) - df_combined['Stroop_Test']

# Now the range will be inverted.
print(f"New range of Stroop_Test: {df_combined['Stroop_Test'].min()} - {df_combined['Stroop_Test'].max()}")

# plot histogram of transformed Stroop_Test
plt.figure(figsize=(8, 6))
sns.histplot(df_combined['Stroop_Test'].dropna(), bins=30, kde=True)
plt.title('Histogram of Transformed Stroop_Test')
plt.xlabel('Transformed Stroop_Test Score')
plt.ylabel('Frequency')
plt.show()

# %%
# save back
# df_combined.to_csv(data_save_path + 'VETSA_dataset_renamed.csv')
