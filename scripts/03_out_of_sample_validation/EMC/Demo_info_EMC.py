import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')  # path to the utils.py file
import utils

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# %%
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'  # where the data is saved
results_path = f'/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results/AutoGluon/EMC/{target}/{feature_comb}/'  # where the results will be saved


def describe_and_count(df):
    """
    Describe key numeric columns and count SEX/APOE4 categories.
    Safely skips any columns that are missing.
    """
    df = df.copy()

    # --- numeric summaries ---
    numeric_cols = [
        'Age_at_Scan',
        'Depression_score',
        'BMI',
        'PSG_Sleep_Dur',
        'Self_Sleep_Dur',
        'PSG_Sleep_Eff',
        'Self_Sleep_Eff',
    ]
    existing_numeric = [c for c in numeric_cols if c in df.columns]
    missing_numeric = [c for c in numeric_cols if c not in df.columns]

    if existing_numeric:
        print("Numeric variable summary:")
        print(df[existing_numeric].describe())
    else:
        print(f"No numeric columns present among: {numeric_cols}")

    if missing_numeric:
        print("Missing numeric columns:", ", ".join(missing_numeric))

    # --- SEX counts ---
    if 'SEX' in df.columns:
        df.loc[:, 'SEX'] = df['SEX'].astype('category')
        sex_count = df['SEX'].value_counts().get(2, 0)
        print(f"Number of subjects with SEX value 2: {sex_count}")
    else:
        print("Column 'SEX' not found; skipping SEX summary.")

    # --- APOE4 counts ---
    if 'APOE4' in df.columns:
        apoe_counts = df['APOE4'].value_counts()
        apoe4_count_1 = apoe_counts.get(1, 0)
        apoe4_count_2 = apoe_counts.get(2, 0)
        apoe4_missing_count = df['APOE4'].isna().sum()

        print(f"Number of subjects with APOE4 value 1: {apoe4_count_1}")
        print(f"Number of subjects with APOE4 value 2: {apoe4_count_2}")
        print(f"Number of missing values in APOE4 column: {apoe4_missing_count}")
    else:
        print("Column 'APOE4' not found; skipping APOE4 counts.")


df_EMC = pd.read_csv(os.path.join(data_save_path, 'EMC.csv'))  # EMC csv file name
df_EMC = df_EMC.dropna(subset=['Stroop_Test'])

# --- capture output of describe_and_count and save to TXT ---
import io
import contextlib

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    describe_and_count(df_EMC)

output_text = buffer.getvalue()

# Make sure results directory exists
os.makedirs(results_path, exist_ok=True)
out_file = os.path.join(results_path, "EMC_describe_and_count.txt")

with open(out_file, "w") as f:
    f.write(output_text)

# Also show the same text in the console once
print(output_text)
print(f"\nSummary saved to: {out_file}")
