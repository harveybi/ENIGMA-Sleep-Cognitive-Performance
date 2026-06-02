import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils
import ml_pipeline
from ml_pipeline import MLPipeline

import argparse
import pandas as pd

# %%
# reload utils
import importlib
importlib.reload(utils)
importlib.reload(ml_pipeline)
import ml_pipeline
from ml_pipeline import MLPipeline

# %%
# Argument parsing
parser = argparse.ArgumentParser(description='Ridge Regression for n-back working memory in Liege dataset.')
parser.add_argument('feature_comb', type=str, help='''Name of the feature combination to be used.
    Available combinations are:
    - Sleep: Uses sleep features only.
    - Cov: Uses covariates only.
    - Sleep_Cov: Uses sleep features combined with covariates.
    - Sleep_Cov_Brain: Uses sleep features, covariates, and brain imaging data.
    - Sleep_Cov_CT: Uses sleep features, covariates, and cortical thickness from imaging data.
    - Sleep_Cov_SA: Uses sleep features, covariates, and surface area from imaging data.
    - Sleep_Cov_Subcor: Uses sleep features, covariates, and subcortical volumes.
    - Cov_Brain: Uses covariates combined with brain imaging data.
    - Cov_CT: Uses covariates and cortical thickness from imaging data.
    - Cov_SA: Uses covariates and surface area from imaging data.
    - Cov_Subcor: Uses covariates and subcortical volumes.
    - Sleep_Shuffle_Cov: Uses sleep features (shuffled) and covariates.
    - Sleep_Cov_Subcor_Shuffle: Uses sleep features, covariates, and subcortical volumes (shuffled).
    - Cov_Brain_Shuffle: Uses covariates and brain imaging data (shuffled).
    - Cov_Subcor_Shuffle: Uses covariates and subcortical volumes (shuffled).
''')

parser.add_argument('target', type=str, help='''Name of the target to be predicted.
    Available targets are:
    - Stroop: Stroop_Test
    - Memory: Average n-back working memory accuracy
    - 1-back: 1-back working memory accuracy
    - 2-back: 2-back working memory accuracy
    - 3-back: 3-back working memory accuracy
''')

# Add num_cores as an optional argument with a default value
parser.add_argument('num_cores', type=int, default=1,
                    help='Number of cores to be used for parallel processing. Default is 1.')

args = parser.parse_args()
feature_comb = args.feature_comb
target = args.target
num_cores = args.num_cores

print(f"\nStarting Ridge regression pipeline for {target} prediction with feature combination {feature_comb}.\n")

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/Ridge/{target}/Liege/'

# %%
df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed.csv')
df_Liege_cleaned = pd.read_csv(data_save_path + 'Liege_dataset_renamed_cleaned.csv')

# %%
"""
Define feature lists
"""
Sleep = ['PSG_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Dur', 'Self_Sleep_Eff', 'Depression_BDII']
Cov = ['Age_at_Scan', 'SEX', 'BMI']
TIV = 'EstimatedTotalIntraCranialVol'

columns = df_Liege.columns.tolist()
Thickness_DK = columns[columns.index('lh_bankssts_thickness'):columns.index('rh_insula_thickness')+1]
Thickness_Schaefer = columns[columns.index('LH_Vis_1_thickness'):columns.index('RH_Default_pCunPCC_9_thickness')+1]
Area_DK = columns[columns.index('lh_bankssts_area'):columns.index('rh_insula_area')+1]
Area_Schaefer = columns[columns.index('LH_Vis_1_area'):columns.index('RH_Default_pCunPCC_9_area')+1]
Subcortical = columns[columns.index('Left-Lateral-Ventricle'):columns.index('CC_Anterior')+1]

targets = ['Stroop_Test', 'Memory_Test']

# %%
"""
Data preprocessing for Liege dataset
1. Convert sleep measurements units
2. Remove subjects with missing data in necessary features
3. Outlier detection and removal
4. Add group based on age and sex
5. Brain correction by brain size using internal data normalisation
6. NAI_Wordlist_Test transfer to accuracy
"""
# convert sleep measurements units
sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']
df_Liege_ml = utils.convert_units(df_Liege, sleep_dur_cols, sleep_eff_cols)

# remove subjects with missing data in necessary features
important_variables_list = Sleep + Cov + [TIV] + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical + targets
df_Liege_ml_cld = utils.clean_missing_data(df_Liege_ml, important_variables_list)

# outlier detection and removal
df_Liege_ml_cld_out = utils.rm_outliers(df_Liege_ml_cld, important_variables_list)

# add a column of 'Age_Group' after 'Age_at_Scan'.
df_Liege_ml_cld_out = utils.add_age_groups(df_Liege_ml_cld_out)

# add a column of Group which represents the group of age groups and SEX groups
df_Liege_ml_cld_out = utils.add_groups_age_sex(df_Liege_ml_cld_out)

print("Data type of SEX column before conversion:", df_Liege_ml_cld_out['SEX'].dtype)
# Convert to int if it's not already an int
if df_Liege_ml_cld_out['SEX'].dtype != 'int64':
    df_Liege_ml_cld_out['SEX'] = df_Liege_ml_cld_out['SEX'].astype(int)
print("Data type of SEX column after conversion:", df_Liege_ml_cld_out['SEX'].dtype)

# %%
# Brain correction by brain size using internal data normalisation
df_Liege_ml_cld_out[Thickness_DK] = df_Liege_ml_cld_out[Thickness_DK].div(df_Liege_ml_cld_out[Thickness_DK].sum(axis=1), axis=0)
df_Liege_ml_cld_out[Thickness_Schaefer] = df_Liege_ml_cld_out[Thickness_Schaefer].div(df_Liege_ml_cld_out[Thickness_Schaefer].sum(axis=1), axis=0)
df_Liege_ml_cld_out[Area_DK] = df_Liege_ml_cld_out[Area_DK].div(df_Liege_ml_cld_out[Area_DK].sum(axis=1), axis=0)
df_Liege_ml_cld_out[Area_Schaefer] = df_Liege_ml_cld_out[Area_Schaefer].div(df_Liege_ml_cld_out[Area_Schaefer].sum(axis=1), axis=0)
df_Liege_ml_cld_out[Subcortical] = df_Liege_ml_cld_out[Subcortical].div(df_Liege_ml_cld_out['EstimatedTotalIntraCranialVol'], axis=0)


# %%
def check_normalisation(sub):
    sum_thickness_DK = df_Liege_ml_cld_out[Thickness_DK].iloc[sub].sum(axis=0)
    sum_thickness_Schaefer = df_Liege_ml_cld_out[Thickness_Schaefer].iloc[sub].sum(axis=0)
    sum_area_DK = df_Liege_ml_cld_out[Area_DK].iloc[sub].sum(axis=0)
    sum_area_Schaefer = df_Liege_ml_cld_out[Area_Schaefer].iloc[sub].sum(axis=0)
    sum_subcortical = df_Liege_ml_cld_out[Subcortical].iloc[sub].sum(axis=0)

    print(f"Sum of thickness_DK: {sum_thickness_DK}")
    print(f"Sum of thickness_Schaefer: {sum_thickness_Schaefer}")
    print(f"Sum of area_DK: {sum_area_DK}")
    print(f"Sum of area_Schaefer: {sum_area_Schaefer}")
    print(f"Sum of subcortical: {sum_subcortical}")


# %%
"""
Define X and y
"""
X_dict = {
    'Sleep': Sleep,
    'Cov': Cov,
    'Sleep_Cov': Sleep + Cov,
    'Sleep_Cov_Brain': Sleep + Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Sleep_Cov_CT': Sleep + Cov + Thickness_DK + Thickness_Schaefer,
    'Sleep_Cov_SA': Sleep + Cov + Area_DK + Area_Schaefer,
    'Sleep_Cov_Subcor': Sleep + Cov + Subcortical,
    'Cov_Brain': Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Cov_CT': Cov + Thickness_DK + Thickness_Schaefer,
    'Cov_SA': Cov + Area_DK + Area_Schaefer,
    'Cov_Subcor': Cov + Subcortical,
    'Sleep_Shuffle_Cov': Sleep + Cov,
    'Sleep_Cov_Subcor_Shuffle': Sleep + Cov + Subcortical,
    'Cov_Brain_Shuffle': Cov + Thickness_DK + Thickness_Schaefer + Area_DK + Area_Schaefer + Subcortical,
    'Cov_Subcor_Shuffle': Cov + Subcortical
}

y_dict = {
    'Stroop': 'Stroop_Test',
    'Memory': 'Memory_Test'
}

# %%
print(f"\nStart model training")
case_results_path = results_path + f"{feature_comb}/"
if not os.path.exists(case_results_path):
    os.makedirs(case_results_path)

X = X_dict[feature_comb]
y = y_dict[target]
df_train = df_Liege_ml_cld_out
df_test = None
save_path = case_results_path

ml_pipe_instance = MLPipeline(X, y, df_train, df_test, num_cores, save_path)
scores, model = ml_pipe_instance.ridge_rg_pipe(stratified_label='Group_Age_SEX')

print(f"Model training completed.\n")

# %%
# save scores and model
scores.to_csv(case_results_path + 'scores.csv')
from joblib import dump
dump(model, case_results_path + 'model.joblib')

# %%
print("\nModel performance:")
print("======================================")
print(f"Average CV Train MAE: {-scores['train_neg_mean_absolute_error'].mean():.4f}")
print(f"Average CV Test MAE: {-scores['test_neg_mean_absolute_error'].mean():.4f}")
print(f"Average CV Train RMSE: {-scores['train_neg_root_mean_squared_error'].mean():.4f}")
print(f"Average CV Test RMSE: {-scores['test_neg_root_mean_squared_error'].mean():.4f}")
print(f"Average CV Train R2: {scores['train_r2'].mean():.4f}")
print(f"Average CV Test R2: {scores['test_r2'].mean():.4f}")
print(f"Average CV Train Pearson r: {scores['train_r_corr'].mean():.4f}")
print(f"Average CV Test Pearson r: {scores['test_r_corr'].mean():.4f}")
print("======================================\n")

# create a dataframe to store the results
results_df = pd.DataFrame({
    'Model': ['Ridge Regression'],
    'Feature Combination': [feature_comb],
    'Target': [target],
    'Average CV Train MAE': [-scores['train_neg_mean_absolute_error'].mean()],
    'Average CV Test MAE': [-scores['test_neg_mean_absolute_error'].mean()],
    'Average CV Train RMSE': [-scores['train_neg_root_mean_squared_error'].mean()],
    'Average CV Test RMSE': [-scores['test_neg_root_mean_squared_error'].mean()],
    'Average CV Train R2': [scores['train_r2'].mean()],
    'Average CV Test R2': [scores['test_r2'].mean()],
    'Average CV Train Pearson r': [scores['train_r_corr'].mean()],
    'Average CV Test Pearson r': [scores['test_r_corr'].mean()]
})

results_df.to_csv(case_results_path + 'results.csv')

# %%
feature_list = X_dict[feature_comb] + [y_dict[target]] + ['Stroop_Test']
utils.plot_corr_matrix(df_Liege_ml_cld_out, feature_list, 'Liege')

# %%
# distribution plot of the target, use seaborn
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df_Liege_ml_cld_out[y_dict[target]])
plt.xlabel(y_dict[target])
plt.ylabel('Frequency')
plt.title(f'Distribution of {y_dict[target]} in Liege dataset')
# plt.savefig(case_results_path + f'{target}_distribution.png')
plt.show()

sns.histplot(df_Liege_ml_cld_out['Stroop_Test'])
plt.xlabel(y_dict[target])
plt.ylabel('Frequency')
plt.title(f'Distribution of Stroop_Test in Liege dataset')
# plt.savefig(case_results_path + f'{target}_distribution.png')
plt.show()

sns.histplot(df_Liege_ml_cld_out['Age_at_Scan'])
plt.xlabel(y_dict[target])
plt.ylabel('Frequency')
plt.title(f'Distribution of Age_at_Scan in Liege dataset')
# plt.savefig(case_results_path + f'{target}_distribution.png')
plt.show()

# %%
# make a dataframe only include 'Sub_ID' and target
df_target = df_Liege_ml_cld_out[['Sub_ID', y_dict[target], 'Memory_Test.2']]
