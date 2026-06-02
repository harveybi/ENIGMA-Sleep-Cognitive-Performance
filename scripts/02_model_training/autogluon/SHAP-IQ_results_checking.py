import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')
import utils

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import julearn
import itertools
import starbars

from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
)
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

# %%
# ------------------------------- configuration -------------------------------
base_dir = Path("/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Results/AutoGluon")

feature_list = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
                'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
                'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE',
                'Sleep_Cov_APOE_Shuffle', 'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle']

target_list  = ['Stroop', 'Memory', 'Stroop_rgo_age', 'Memory_rgo_age']

# %%
for target, feature in itertools.product(target_list, feature_list):
    # if target has '_rgo_age' in target, only work with 'Sleep_Cov' and 'Sleep_Cov_Brain'
    if '_rgo_age' in target:
        if feature not in ['Sleep_Cov', 'Sleep_Cov_Brain']:
            continue

    file_path = base_dir / target / "SHIP_Trend" / feature / "shapiq" / "ivs_SHIP_SHIP.pkl"

    if not file_path.exists():
        print(f"File {file_path} does not exist.")
        continue
