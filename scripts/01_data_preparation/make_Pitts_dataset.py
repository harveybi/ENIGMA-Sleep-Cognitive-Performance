import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
templates_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/data_extract_scripts/'

# %%
# Load the data
Pitts_data_path = data_save_path + 'raw_datasets/'

df_Pitts_demo, df_Pitts_imaging = utils.dataset_loader(Pitts_data_path + 'Confirmed_ENIGMA-Sleep&Cognition_Phenotypic template_PyNEL.xlsx')
df_Pitts_demo_all = utils.rm_missed_measure(df_Pitts_demo)

df_Pitts_self_sleep = pd.read_excel(Pitts_data_path + 'PyNEL_SelfReport_Sleep.xlsx')

df_Pitts_subcor = pd.read_csv(Pitts_data_path + 'Sorted_PyNeL_subcortical_volume.csv')

# %%
# sort df_Pitts_demo_all, df_Pitts_imaging by 'Sub_ID'
df_Pitts_demo_all = df_Pitts_demo_all.sort_values(by='Sub_ID')
df_Pitts_imaging = df_Pitts_imaging.sort_values(by='Sub_ID')

# sort df_Pitts_self_sleep, df_Pitts_subcor by '8DigID'
df_Pitts_self_sleep = df_Pitts_self_sleep.sort_values(by='8DigID')
df_Pitts_subcor = df_Pitts_subcor.sort_values(by='8DigID')

# %%
# rename the '8DigID' column to 'Sub_ID' in df_Pitts_self_sleep, df_Pitts_subcor
df_Pitts_self_sleep = df_Pitts_self_sleep.rename(columns={'8DigID': 'Sub_ID'})
df_Pitts_subcor = df_Pitts_subcor.rename(columns={'8DigID': 'Sub_ID'})

# drop 'TimePoint' column in df_Pitts_subcor
df_Pitts_subcor = df_Pitts_subcor.drop(columns=['TimePoint'])

# %%
# add 'Depression_score' column in df_Pitts_demo_all. value are nan
df_Pitts_demo_all['Depression_score'] = [float('nan')] * len(df_Pitts_demo_all)

# %%
# in df_Pitts_demo_all, add 'Self_Sleep_Dur', 'Self_Sleep_Eff' columns. They are from df_Pitts_self_sleep 'Self Report Total Sleep Time', 'Self Report Sleep Efficiency'
df_Pitts_demo_all['Self_Sleep_Dur'] = df_Pitts_self_sleep['Self Report Total Sleep Time']
df_Pitts_demo_all['Self_Sleep_Eff'] = df_Pitts_self_sleep['Self Report Sleep Efficiency']

# %%
# rename subcortical columns
label_subcortical = ['Sub_ID', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
                     'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
                     '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
                     'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel', 'Right-Lateral-Ventricle',
                     'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
                     'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
                     'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel', '5th-Ventricle',
                     'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
                     'EstimatedTotalIntraCranialVol']
df_Pitts_subcor = df_Pitts_subcor[label_subcortical]

# rename 'Left-Thalamus-Proper', 'Right-Thalamus-Proper' to 'Left-Thalamus', 'Right-Thalamus'; 'subject_ID' to 'Sub_ID'
df_Pitts_subcor = df_Pitts_subcor.rename(columns={'Left-Thalamus-Proper': 'Left-Thalamus',
                                                  'Right-Thalamus-Proper': 'Right-Thalamus'})

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
# sort these three dataframes by 'Sub_ID'
df_Pitts_demo_all = df_Pitts_demo_all.sort_values(by='Sub_ID')
df_Pitts_imaging = df_Pitts_imaging.sort_values(by='Sub_ID')
df_Pitts_subcor = df_Pitts_subcor.sort_values(by='Sub_ID')

# %%
# unify the datatype of 'Sub_ID' in these three dataframes
df_Pitts_demo_all['Sub_ID'] = df_Pitts_demo_all['Sub_ID'].astype(str)
df_Pitts_imaging['Sub_ID'] = df_Pitts_imaging['Sub_ID'].astype(str)
df_Pitts_subcor['Sub_ID'] = df_Pitts_subcor['Sub_ID'].astype(str)

df_Pitts_demo_all = df_Pitts_demo_all.set_index('Sub_ID')
df_Pitts_imaging = df_Pitts_imaging.set_index('Sub_ID')
df_Pitts_subcor = df_Pitts_subcor.set_index('Sub_ID')

# %%
# concatenate the dataframes, there are subjects that are not in all three dataframes
df_Pitts = pd.concat([df_Pitts_demo_all, df_Pitts_imaging, df_Pitts_subcor], axis=1, join='inner')

# %%
X_Thickness_DK = df_Pitts_imaging.columns[0:68].tolist()
X_Thickness_Schaefer = df_Pitts_imaging.columns[68:468].tolist()
X_Area_DK = df_Pitts_imaging.columns[468:536].tolist()
X_Area_Schaefer = df_Pitts_imaging.columns[536:].tolist()

# Creating rename mapping dictionaries
rename_mapping_dk_ct = dict(zip(X_Thickness_DK, renamed_dk_ct_df_columns))
rename_mapping_schaefer_ct = dict(zip(X_Thickness_Schaefer, renamed_schaefer_ct_df_columns))
rename_mapping_dk_sa = dict(zip(X_Area_DK, renamed_dk_sa_df_columns))
rename_mapping_schaefer_sa = dict(zip(X_Area_Schaefer, renamed_schaefer_sa_df_columns))

# Combining all rename mapping dictionaries into one
all_rename_mappings = {**rename_mapping_dk_ct, **rename_mapping_schaefer_ct,
                       **rename_mapping_dk_sa, **rename_mapping_schaefer_sa}

# Renaming the columns in df_Pitts_ml
df_Pitts_renamed = df_Pitts.rename(columns=all_rename_mappings)

# %%
# Save the data
# df_Pitts_renamed.to_csv(data_save_path + 'Pitts_dataset_renamed.csv')

# %%
print(f"Range of Stroop_Test: {df_Pitts_renamed['Executive_Functioning'].min()} - {df_Pitts_renamed['Executive_Functioning'].max()}")

# plot the distribution of 'Executive_Functioning'
plt.figure(figsize=(8, 6))
sns.histplot(df_Pitts_renamed['Executive_Functioning'], bins=30, kde=True)
plt.title('Distribution of Executive_Functioning (Stroop Test)')
plt.xlabel('Executive_Functioning Score')
plt.ylabel('Frequency')
plt.show()

# %%
# Inverse Transformation (Simple Subtraction) for 'Executive_Functioning'. Transformed Score=Max−Original Score
df_Pitts_renamed['Executive_Functioning'] = max(df_Pitts_renamed['Executive_Functioning']) - (df_Pitts_renamed['Executive_Functioning'] - min(df_Pitts_renamed['Executive_Functioning']))

print(f"New Range of Stroop_Test: {df_Pitts_renamed['Executive_Functioning'].min()} - {df_Pitts_renamed['Executive_Functioning'].max()}")

# plot the new distribution of 'Executive_Functioning'
plt.figure(figsize=(8, 6))
sns.histplot(df_Pitts_renamed['Executive_Functioning'], bins=30, kde=True)
plt.title('New Distribution of Executive_Functioning (Stroop Test)')
plt.xlabel('Executive_Functioning Score')
plt.ylabel('Frequency')
plt.show()

# %%
# transform 'Memory_Test1' by /32 * 100
df_Pitts_renamed['Memory_Test1'] = (df_Pitts_renamed['Memory_Test1'] / 32) * 100

# %%
# transform 'Memory_Test2' by /21 * 100
df_Pitts_renamed['Memory_Test2'] = (df_Pitts_renamed['Memory_Test2'] / 21) * 100

# %%
# save the transformed data
# df_Pitts_renamed.to_csv(data_save_path + 'Pitts_dataset_renamed_target_transformed.csv')
