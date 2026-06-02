import os
import sys

sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils

import pandas as pd

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'

# %%
# Load the data
KI_data_path = data_save_path + 'raw_datasets/'

df_KI_demo, df_KI_imaging = utils.dataset_loader(KI_data_path + 'ENIGMA-Sleep&Cognition_Phenotypic_v4__HB_Jul06.csv')
df_KI_demo_all = utils.rm_missed_measure(df_KI_demo)

df_KI_subcor = pd.read_csv(KI_data_path + 'KI_subcor_volume_filtered.csv')
# rename the 'Sub_ID' column's value from 'sub-9001' to '9001'
df_KI_subcor['Sub_ID'] = df_KI_subcor['Sub_ID'].apply(lambda x: x.split('-')[1])

# %%
# rename columns
label_subcortical = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
                     'Left-Cerebellum-Cortex', 'Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
                     '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
                     'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel', 'Right-Lateral-Ventricle',
                     'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex', 'Right-Thalamus',
                     'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
                     'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel', '5th-Ventricle', 'CC_Posterior',
                     'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
                     'EstimatedTotalIntraCranialVol']
df_KI_subcor.columns = ['Sub_ID'] + label_subcortical

# rename cortical thickness
# Schaefer
example_schaefer_ct_df = pd.read_csv(
    '/data/project/sleep_ENIGMA_insomnia/Codes/data_extract_scripts/control_thickness_Schaefer.csv')
filtered_columns = [col for col in example_schaefer_ct_df.columns if
                    col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV']]
renamed_schaefer_ct_df_columns = [col.replace('lh_7Networks_', '').replace('rh_7Networks_', '') for col in
                                  filtered_columns]
# DK
example_dk_ct_df = pd.read_csv(
    '/data/project/sleep_ENIGMA_insomnia/Codes/data_extract_scripts/control_thickness_DK.csv')
renamed_dk_ct_df_columns = [col for col in example_dk_ct_df.columns if
                            col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV']]

# rename surface area
# Schaefer
example_schaefer_sa_df = pd.read_csv(
    '/data/project/sleep_ENIGMA_insomnia/Codes/data_extract_scripts/control_area_Schaefer.csv')
filtered_columns = [col for col in example_schaefer_sa_df.columns if
                    col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV', 'lh_WhiteSurfArea_area',
                                'rh_WhiteSurfArea_area']]
renamed_schaefer_sa_df_columns = [col.replace('lh_7Networks_', '').replace('rh_7Networks_', '') for col in
                                  filtered_columns]
# DK
example_dk_sa_df = pd.read_csv('/data/project/sleep_ENIGMA_insomnia/Codes/data_extract_scripts/control_area_DK.csv')
renamed_dk_sa_df_columns = [col for col in example_dk_sa_df.columns if
                            col not in ['subject_ID', 'BrainSegVolNotVent', 'eTIV']]

# %%
# sort these three dataframes by 'Sub_ID'
df_KI_demo_all = df_KI_demo_all.sort_values(by='Sub_ID')
df_KI_imaging = df_KI_imaging.sort_values(by='Sub_ID')
df_KI_subcor = df_KI_subcor.sort_values(by='Sub_ID')

# %%
# print the datatype of 'Sub_ID' in these three dataframes
print(df_KI_demo_all['Sub_ID'].dtype)
print(df_KI_imaging['Sub_ID'].dtype)
print(df_KI_subcor['Sub_ID'].dtype)

# %%
# unify the datatype of 'Sub_ID' in these three dataframes
df_KI_demo_all['Sub_ID'] = df_KI_demo_all['Sub_ID'].astype(str)
df_KI_imaging['Sub_ID'] = df_KI_imaging['Sub_ID'].astype(str)
df_KI_subcor['Sub_ID'] = df_KI_subcor['Sub_ID'].astype(str)

# %%
df_KI_demo_all = df_KI_demo_all.set_index('Sub_ID')
df_KI_imaging = df_KI_imaging.set_index('Sub_ID')
df_KI_subcor = df_KI_subcor.set_index('Sub_ID')

# %%
# concatenate the dataframes, there are subjects that are not in all three dataframes
df_KI = pd.concat([df_KI_demo_all, df_KI_imaging, df_KI_subcor], axis=1, join='inner')

# %%
X_Thickness_DK = df_KI_imaging.columns[0:68].tolist()
X_Thickness_Schaefer = df_KI_imaging.columns[68:468].tolist()
X_Area_DK = df_KI_imaging.columns[468:536].tolist()
X_Area_Schaefer = df_KI_imaging.columns[536:].tolist()

# Creating rename mapping dictionaries
rename_mapping_dk_ct = dict(zip(X_Thickness_DK, renamed_dk_ct_df_columns))
rename_mapping_schaefer_ct = dict(zip(X_Thickness_Schaefer, renamed_schaefer_ct_df_columns))
rename_mapping_dk_sa = dict(zip(X_Area_DK, renamed_dk_sa_df_columns))
rename_mapping_schaefer_sa = dict(zip(X_Area_Schaefer, renamed_schaefer_sa_df_columns))

# Combining all rename mapping dictionaries into one
all_rename_mappings = {**rename_mapping_dk_ct, **rename_mapping_schaefer_ct,
                       **rename_mapping_dk_sa, **rename_mapping_schaefer_sa}

# Renaming the columns in df_KI_ml
df_KI_renamed = df_KI.rename(columns=all_rename_mappings)

# %%
# rename 'Depression_HADS' column to 'Depression_score'
df_KI_renamed = df_KI_renamed.rename(columns={'Depression_HADS': 'Depression_score'})

# 'Memory_Test' values * 100
df_KI_renamed['Memory_Test'] = df_KI_renamed['Memory_Test'] * 100


# %%
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


# use this function to convert 'Depression_score' column's values
df_KI_renamed['Depression_score'] = df_KI_renamed['Depression_score'].apply(hads_to_bdi)

# %%
# Save the data
df_KI_renamed.to_csv(data_save_path + 'KI_dataset_renamed.csv')

# %%
# convert sleep measurements units
sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff']
df_KI_ml = utils.convert_units(df_KI_renamed, sleep_dur_cols, sleep_eff_cols)

# remove subjects with missing data in necessary features
X_sleep = ['PSG_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Dur', 'Depression_score']
X_cov = ['Age_at_Scan', 'SEX', 'BMI']
X_list = X_sleep + X_cov
y_list = ['Memory_Test']
Brain_img = df_KI_renamed.columns[15:]

features = X_list + Brain_img.tolist() + y_list

df_KI_ml_cld = utils.clean_missing_data(df_KI_ml, features)

# outlier detection and removal
df_KI_ml_cld_out = utils.rm_outliers(df_KI_ml_cld, features)

# %%
# save the cleaned data
df_KI_ml_cld_out.to_csv(data_save_path + 'KI_dataset_cleaned.csv')
