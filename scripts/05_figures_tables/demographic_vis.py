import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# reload utils
import importlib
importlib.reload(utils)

# %%
# Load data
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'

df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed_cleaned.csv')
df_Liege = pd.read_csv(data_save_path + 'Liege_dataset_renamed_cleaned.csv')
df_KI = pd.read_csv(data_save_path + 'KI_dataset_renamed_cleaned.csv')

# %%
# larger figure size
plt.figure(figsize=(10, 6))
sns.histplot(data=df_SHIP, x="Stroop_Test", color=sns.color_palette()[0], label="SHIP_Stroop", kde=True)
sns.histplot(data=df_Liege, x="Stroop_Test", color=sns.color_palette()[1], label="Liege_Stroop", kde=True)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# %%
sns.histplot(data=df_SHIP, x="Memory_Test", color=sns.color_palette()[0], label="SHIP_Memory", kde=True)
sns.histplot(data=df_Liege, x="Memory_Test", color=sns.color_palette()[1], label="Liege_Memory", kde=True)
sns.histplot(data=df_KI, x="Memory_Test", color=sns.color_palette()[2], label="KI_Memory", kde=True)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# %%
sns.histplot(data=df_SHIP, x="Age_at_Scan", color=sns.color_palette()[0], label="SHIP_Age", kde=True)
sns.histplot(data=df_Liege, x="Age_at_Scan", color=sns.color_palette()[1], label="Liege_Age", kde=True)
sns.histplot(data=df_KI, x="Age_at_Scan", color=sns.color_palette()[2], label="KI_Age", kde=True)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# %%
# correlation matrix, df, feature_list, site_name
SHIP_feature_list = ['Age_at_Scan', 'SEX', 'BMI', 'Depression_score', 'PSG_Sleep_Dur', 'Self_Sleep_Dur',
                     'PSG_Sleep_Eff', 'Self_Sleep_Eff', 'Stroop_Test', 'Memory_Test']
Liege_feature_list = ['Age_at_Scan', 'SEX', 'BMI', 'Depression_score', 'PSG_Sleep_Dur', 'Self_Sleep_Dur',
                      'PSG_Sleep_Eff', 'Self_Sleep_Eff', 'Stroop_Test', 'Memory_Test', '1-back_acc', '2-back_acc',
                      '3-back_acc']
KI_feature_list = ['Age_at_Scan', 'SEX', 'BMI', 'Depression_score', 'PSG_Sleep_Dur', 'Self_Sleep_Dur', 'PSG_Sleep_Eff',
                   'Memory_Test']
utils.plot_corr_matrix(df_SHIP, SHIP_feature_list, 'SHIP')
utils.plot_corr_matrix(df_Liege, Liege_feature_list, 'Liege')
utils.plot_corr_matrix(df_KI, KI_feature_list, 'KI')
