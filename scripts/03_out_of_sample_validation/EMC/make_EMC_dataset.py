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
EMC_data_path = data_save_path

df_EMC_demo, df_EMC_imaging = utils.dataset_loader(EMC_data_path + '???.csv')  # edit to your data file name
df_EMC_demo_all = utils.rm_missed_measure(df_EMC_demo)

df_EMC_subcor_raw = pd.read_csv(EMC_data_path + '???.csv')  # edit to your data file name

# %%
# rename columns
label_subcortical = ['subject_ID', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
                     'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
                     '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
                     'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel', 'Right-Lateral-Ventricle',
                     'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
                     'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
                     'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel', '5th-Ventricle',
                     'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
                     'EstimatedTotalIntraCranialVol']
df_EMC_subcor = df_EMC_subcor_raw[label_subcortical]

# %%
# rename 'Left-Thalamus-Proper', 'Right-Thalamus-Proper' to 'Left-Thalamus', 'Right-Thalamus'; 'subject_ID' to 'Sub_ID'
df_EMC_subcor = df_EMC_subcor.rename(columns={'Left-Thalamus-Proper': 'Left-Thalamus',
                                              'Right-Thalamus-Proper': 'Right-Thalamus',
                                              'subject_ID': 'Sub_ID'})

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
TODO: Check the Sub_ID in df_EMC_subcor. If the format of Sub_ID in df_EMC_subcor is different from the format of
Sub_ID in df_EMC_demo_all and df_EMC_imaging, unify the format of Sub_ID in these three dataframes.
Example:
df_EMC_subcor['Sub_ID'] = df_EMC_subcor['Sub_ID'].str.replace('sub-', '')
"""

# %%
# sort these three dataframes by 'Sub_ID'
df_EMC_demo_all = df_EMC_demo_all.sort_values(by='Sub_ID')
df_EMC_imaging = df_EMC_imaging.sort_values(by='Sub_ID')
df_EMC_subcor = df_EMC_subcor.sort_values(by='Sub_ID')

# %%
# unify the datatype of 'Sub_ID' in these three dataframes
df_EMC_demo_all['Sub_ID'] = df_EMC_demo_all['Sub_ID'].astype(str)
df_EMC_imaging['Sub_ID'] = df_EMC_imaging['Sub_ID'].astype(str)
df_EMC_subcor['Sub_ID'] = df_EMC_subcor['Sub_ID'].astype(str)

# %%
df_EMC_demo_all = df_EMC_demo_all.set_index('Sub_ID')
df_EMC_imaging = df_EMC_imaging.set_index('Sub_ID')
df_EMC_subcor = df_EMC_subcor.set_index('Sub_ID')

# %%
# concatenate the dataframes, there are subjects that are not in all three dataframes
df_EMC = pd.concat([df_EMC_demo_all, df_EMC_imaging, df_EMC_subcor], axis=1, join='inner')

# %%
X_Thickness_DK = df_EMC_imaging.columns[0:68].tolist()
X_Thickness_Schaefer = df_EMC_imaging.columns[68:468].tolist()
X_Area_DK = df_EMC_imaging.columns[468:536].tolist()
X_Area_Schaefer = df_EMC_imaging.columns[536:].tolist()

# Creating rename mapping dictionaries
rename_mapping_dk_ct = dict(zip(X_Thickness_DK, renamed_dk_ct_df_columns))
rename_mapping_schaefer_ct = dict(zip(X_Thickness_Schaefer, renamed_schaefer_ct_df_columns))
rename_mapping_dk_sa = dict(zip(X_Area_DK, renamed_dk_sa_df_columns))
rename_mapping_schaefer_sa = dict(zip(X_Area_Schaefer, renamed_schaefer_sa_df_columns))

# Combining all rename mapping dictionaries into one
all_rename_mappings = {**rename_mapping_dk_ct, **rename_mapping_schaefer_ct,
                       **rename_mapping_dk_sa, **rename_mapping_schaefer_sa}

# Renaming the columns in df_EMC_ml
df_EMC_renamed = df_EMC.rename(columns=all_rename_mappings)

# %%
"""
TODO: rename your depressive score column to 'Depression_score'
Example:
df_EMC_renamed = df_EMC_renamed.rename(columns={'Depression_HADS': 'Depression_score'})
"""

# %%
"""
TODO: If your depressive score is in HADS scale, you can convert it to BDI scale using the following function:
example:
df_EMC_renamed['Depression_score'] = df_EMC_renamed['Depression_score'].apply(hads_to_bdi)
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
df_EMC_renamed.to_csv(data_save_path + 'EMC_dataset_renamed.csv')  # edit to your data file name
