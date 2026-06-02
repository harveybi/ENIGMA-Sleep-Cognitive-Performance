import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/lib')  # edit to your path
import utils

import pandas as pd

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'  # edit to your path
templates_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/data_extract_scripts/'  # edit to your path

# %%
# Load the data
df_Juelich_1 = pd.read_csv(data_save_path + 'Juelich_SomnoSafe_1_target_cleaned.csv', index_col=0)
df_Juelich_2 = pd.read_csv(data_save_path + 'Juelich_SomnoSafe_2_target_cleaned.csv', index_col=0)
df_Juelich_SD = pd.read_csv(data_save_path + 'Juelich_SomnoSafe_SD_target_cleaned.csv', index_col=0)

df_Juelich_Thickness_DK = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/Juelich/Juelich_thickness_DK.csv')
df_Juelich_Thickness_Schaefer = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/Juelich/Juelich_thickness_Schaefer.csv')
df_Juelich_Area_DK = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/Juelich/Juelich_area_DK.csv')
df_Juelich_Area_Schaefer = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/Juelich/Juelich_area_Schaefer.csv')

df_Juelich_subcor_raw = pd.read_csv('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/raw_datasets/Juelich/Juelich_subcortical_volume.csv')  # edit to your data file name

# %%
# rename columns
label_subcortical = ['subject_ID', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
                     'Left-Cerebellum-Cortex', 'Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
                     '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
                     'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel', 'Right-Lateral-Ventricle',
                     'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
                     'Right-Thalamus', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
                     'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel', '5th-Ventricle',
                     'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
                     'EstimatedTotalIntraCranialVol']
df_Juelich_subcor = df_Juelich_subcor_raw[label_subcortical]

# %%
# rename cortical thickness
# Schaefer
example_schaefer_ct_df = pd.read_csv(templates_path + 'control_thickness_Schaefer.csv', index_col=0)
filtered_columns = [col for col in example_schaefer_ct_df.columns if
                    col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV']]
renamed_schaefer_ct_df_columns = [col.replace('lh_7Networks_', '').replace('rh_7Networks_', '') for col in
                                  filtered_columns]
# DK
example_dk_ct_df = pd.read_csv(templates_path + 'control_thickness_DK.csv', index_col=0)
renamed_dk_ct_df_columns = [col for col in example_dk_ct_df.columns if
                            col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV']]

# rename surface area
# Schaefer
example_schaefer_sa_df = pd.read_csv(templates_path + 'control_area_Schaefer.csv', index_col=0)
filtered_columns = [col for col in example_schaefer_sa_df.columns if
                    col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV', 'lh_WhiteSurfArea_area',
                                'rh_WhiteSurfArea_area']]
renamed_schaefer_sa_df_columns = [col.replace('lh_7Networks_', '').replace('rh_7Networks_', '') for col in
                                  filtered_columns]
# DK
example_dk_sa_df = pd.read_csv(templates_path + 'control_area_DK.csv', index_col=0)
renamed_dk_sa_df_columns = [col for col in example_dk_sa_df.columns if
                            col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV']]

# %%
"""
TODO: Check the Sub_ID in df_Juelich_subcor. If the format of Sub_ID in df_Juelich_subcor is different from the format of
Sub_ID in df_Juelich_demo_all and df_Juelich_imaging, unify the format of Sub_ID in these three dataframes.
Example:
df_Juelich_subcor['Sub_ID'] = df_Juelich_subcor['Sub_ID'].str.replace('sub-', '')
"""
# sort these three dataframes by 'Sub_ID'
df_Juelich_1 = df_Juelich_1.sort_values(by='Sub_ID')
df_Juelich_2 = df_Juelich_2.sort_values(by='Sub_ID')
df_Juelich_SD = df_Juelich_SD.sort_values(by='Sub_ID')
df_Juelich_Thickness_DK = df_Juelich_Thickness_DK.sort_values(by='subject_ID')
df_Juelich_Thickness_Schaefer = df_Juelich_Thickness_Schaefer.sort_values(by='subject_ID')
df_Juelich_Area_DK = df_Juelich_Area_DK.sort_values(by='subject_ID')
df_Juelich_Area_Schaefer = df_Juelich_Area_Schaefer.sort_values(by='subject_ID')
# df_Juelich_subcor = df_Juelich_subcor.sort_values(by='Sub_ID')
df_Juelich_subcor = df_Juelich_subcor.sort_values(by='subject_ID')

# %%
# concatenate df_Juelich_Thickness_DK, df_Juelich_Thickness_Schaefer, df_Juelich_Area_DK, df_Juelich_Area_Schaefer by 'subject_ID' as df_Juelich_imaging
df_Juelich_Thickness_DK.drop(columns=['BrainSegVolNotVent', 'eTIV'], inplace=True)
df_Juelich_Thickness_Schaefer.drop(columns=['BrainSegVolNotVent', 'eTIV'], inplace=True)
df_Juelich_Area_DK.drop(columns=['BrainSegVolNotVent', 'eTIV'], inplace=True)
df_Juelich_Area_Schaefer.drop(columns=['BrainSegVolNotVent', 'eTIV'], inplace=True)

# %%
dfs = [
    df_Juelich_Thickness_DK,
    df_Juelich_Thickness_Schaefer,
    df_Juelich_Area_DK,
    df_Juelich_Area_Schaefer,
    df_Juelich_subcor                                      # already has Sub_ID
]

from functools import reduce
df_Juelich_imaging = reduce(
    lambda left, right: pd.merge(left, right, on='subject_ID', how='outer'),
    dfs
)

# %%
df_Juelich_imaging = df_Juelich_imaging.rename(columns={'subject_ID': 'Sub_ID'})

# %%
def _clean_sub_id(df, col='Sub_ID'):
    return df.assign(**{col: df[col].astype(str).str.replace(r'^sub-', '', regex=True)
                                      .str.zfill(3)})   # example padding; adapt if needed

df_Juelich_1  = _clean_sub_id(df_Juelich_1)
df_Juelich_2  = _clean_sub_id(df_Juelich_2)
df_Juelich_SD = _clean_sub_id(df_Juelich_SD)
# %%
df_Juelich_imaging = _clean_sub_id(df_Juelich_imaging)

# %%
# find all columns which include 'mean' or 'Mean' in their name
mean_columns = [col for col in df_Juelich_imaging.columns if 'mean' in col.lower()]
print(mean_columns)

# remove these columns from df_Juelich_imaging
df_Juelich_imaging.drop(columns=mean_columns, inplace=True)

# %%
whitesurf_columns = [col for col in df_Juelich_imaging.columns if 'WhiteSurfArea' in col]
print(whitesurf_columns)

df_Juelich_imaging.drop(columns=whitesurf_columns, inplace=True)

# %%
def add_imaging(main_df, imaging_df):
    """
    Left-join imaging columns onto main_df, preserving all subjects in main_df.
    Duplicate column names (other than 'Sub_ID') are suffixed with '_img'.
    """
    return main_df.merge(imaging_df, on='Sub_ID', how='left')

df_Juelich_1_with_img  = add_imaging(df_Juelich_1,  df_Juelich_imaging)
df_Juelich_2_with_img  = add_imaging(df_Juelich_2,  df_Juelich_imaging)
df_Juelich_SD_with_img = add_imaging(df_Juelich_SD, df_Juelich_imaging)

# %%
# concatenate the dataframes, there are subjects that are not in all three dataframes
df_Juelich_1_all = df_Juelich_1_with_img
df_Juelich_2_all = df_Juelich_2_with_img
df_Juelich_SD_all = df_Juelich_SD_with_img

# %%
X_Thickness_DK = df_Juelich_imaging.columns[1:69].tolist()
X_Thickness_Schaefer = df_Juelich_imaging.columns[69:469].tolist()
X_Area_DK = df_Juelich_imaging.columns[469:537].tolist()
X_Area_Schaefer = df_Juelich_imaging.columns[537:937].tolist()

# Creating rename mapping dictionaries
rename_mapping_dk_ct = dict(zip(X_Thickness_DK, renamed_dk_ct_df_columns))
rename_mapping_schaefer_ct = dict(zip(X_Thickness_Schaefer, renamed_schaefer_ct_df_columns))
rename_mapping_dk_sa = dict(zip(X_Area_DK, renamed_dk_sa_df_columns))
rename_mapping_schaefer_sa = dict(zip(X_Area_Schaefer, renamed_schaefer_sa_df_columns))

# Combining all rename mapping dictionaries into one
all_rename_mappings = {**rename_mapping_dk_ct, **rename_mapping_schaefer_ct,
                       **rename_mapping_dk_sa, **rename_mapping_schaefer_sa}

# %%
# Renaming the columns in df_Juelich_ml
df_Juelich_1_all_renamed = df_Juelich_1_all.rename(columns=all_rename_mappings)
df_Juelich_2_all_renamed = df_Juelich_2_all.rename(columns=all_rename_mappings)
df_Juelich_SD_all_renamed = df_Juelich_SD_all.rename(columns=all_rename_mappings)

# %%
"""
TODO: rename your depressive score column to 'Depression_score'
Example:
df_Juelich_renamed = df_Juelich_renamed.rename(columns={'Depression_HADS': 'Depression_score'})
"""

# %%
"""
TODO: If your depressive score is in HADS scale, you can convert it to BDI scale using the following function:
example:
df_Juelich_renamed['Depression_score'] = df_Juelich_renamed['Depression_score'].apply(hads_to_bdi)
"""


def hads_to_bdi(hads_score):
    if hads_score < 0 or hads_score > 21:
        return "Invalid HADS score"

    if 0 <= hads_score <= 7:
        bdi_score = (hads_score / 7) * 13
    elif 8 <= hads_score <= 10:
        bdi_score = 14 + (hads_score - 8) * (19 - 14) / (10 - 8)
    elif 11 <= hads_score <= 15:
        bdi_score = 20 + (hads_score - 11) * (28 - 20) / (15 - 11)
    else:  # 16 <= hads_score <= 21
        bdi_score = 29 + (hads_score - 16) * (63 - 29) / (21 - 16)

    return round(bdi_score)


# %%
# Save the data
df_Juelich_1_all_renamed.to_csv(data_save_path + 'Juelich_1_all_renamed_target_cleaned.csv')  # edit to your data file name
df_Juelich_2_all_renamed.to_csv(data_save_path + 'Juelich_2_all_renamed_target_cleaned.csv')  # edit to your data file name
df_Juelich_SD_all_renamed.to_csv(data_save_path + 'Juelich_SD_all_renamed_target_cleaned.csv')  # edit to your data file name
