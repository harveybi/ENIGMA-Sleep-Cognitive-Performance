import os
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')
import utils

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")

import matplotlib as mpl
# Tell Matplotlib: “Whenever I ask for sans-serif, try Arial first”
mpl.rcParams['font.family']      = 'sans-serif'
mpl.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']

from julearn.stats.corrected_ttest import corrected_ttest
from julearn.utils import _compute_cvmdsum

from sklearn.model_selection import (
    KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, check_cv
)

import warnings

# %%
# ------------------------------------------------------------------
# Raw data & CV reconstruction (for missing n_train/n_test)
# ------------------------------------------------------------------
def try_build_ml_dataframe(raw_csv_path: str) -> pd.DataFrame:
    """
    Load the raw CSV and try to reconstruct the ML dataframe used for CV sizing.
    If the user's 'utils' is available (with convert/add_* functions), use it;
    otherwise, return the dataframe as-is.
    """
    df = pd.read_csv(raw_csv_path)
    try:
        import utils  # noqa: F401
        try:
            sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
            sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']
            df = utils.convert_units(df, sleep_dur_cols, sleep_eff_cols)
        except Exception as e:
            warnings.warn(f"[CV sizes] utils.convert_units failed: {e}")

        try:
            df = utils.add_age_groups(df)
        except Exception as e:
            warnings.warn(f"[CV sizes] utils.add_age_groups failed: {e}")

        try:
            df = utils.add_groups_age_sex(df)
        except Exception as e:
            warnings.warn(f"[CV sizes] utils.add_groups_age_sex failed: {e}")
    except Exception as e:
        warnings.warn(f"[CV sizes] 'utils' not available or failed to import: {e}. Proceeding without transformations.")
    return df

def generate_kfold(df_train: pd.DataFrame, y: str | None = None,
                   n_splits: int = 5, random_state: int = 0,
                   stratified: bool = False, n_repeats: int = 1):
    X_data = df_train
    y_data = df_train[y] if (y is not None and y in df_train.columns) else None

    if stratified and (y is not None) and (y_data is not None):
        if n_repeats > 1:
            kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kf.get_n_splits(X_data, y_data)
        return [[train_index, test_index] for train_index, test_index in kf.split(X_data, y_data)]
    else:
        if n_repeats > 1:
            kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kf.get_n_splits(X_data)
        return [[train_index, test_index] for train_index, test_index in kf.split(X_data)]

def compute_cv_sizes_from_raw() -> tuple[int, int] | None:
    """
    Try to rebuild CV splits from the raw dataset and return scalar (n_train, n_test).
    If fold sizes differ (due to remainder splits), we return the *mode*; if no unique
    mode exists, we return the *median* rounded to int.
    """
    raw_csv_path = os.path.join(RAW_DATA_DIR, RAW_DATA_CSV)
    if not os.path.exists(raw_csv_path):
        warnings.warn(f"[CV sizes] Raw CSV not found: {raw_csv_path}")
        return None

    df_ml = try_build_ml_dataframe(raw_csv_path)

    strat_ok = USE_STRAT_CV and (GROUP_LABEL in df_ml.columns) and (df_ml[GROUP_LABEL].notna().any())
    splits = generate_kfold(
        df_ml, y=(GROUP_LABEL if strat_ok else None),
        n_splits=CV_N_SPLITS,
        random_state=CV_RANDOM_SEED,
        stratified=strat_ok,
        n_repeats=CV_N_REPEATS
    )

    ntr = [len(tr) for tr, te in splits]
    nte = [len(te) for tr, te in splits]

    def _to_scalar(seq):
        if not seq:
            return None
        # mode if possible
        try:
            from statistics import mode
            return int(mode(seq))
        except Exception:
            # median fallback
            import math
            return int(math.floor(np.median(seq)))

    ntr_s = _to_scalar(ntr)
    nte_s = _to_scalar(nte)
    if ntr_s is None or nte_s is None:
        return None
    return ntr_s, nte_s

def _build_cv_object(n_splits: int, n_repeats: int, random_state: int, stratified: bool):
    """Return a scikit-learn CV splitter object matching your generate_kfold config."""
    if stratified:
        if n_repeats > 1:
            return RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
        else:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        if n_repeats > 1:
            return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def compute_cv_mdsum_from_raw() -> float | None:
    """
    Rebuild the CV splitter from the raw dataset and compute cv_mdsum
    compatible with julearn's corrected_ttest.
    """
    raw_csv_path = os.path.join(RAW_DATA_DIR, RAW_DATA_CSV)
    if not os.path.exists(raw_csv_path):
        warnings.warn(f"[CV mdsum] Raw CSV not found: {raw_csv_path}")
        return None

    df_ml = try_build_ml_dataframe(raw_csv_path)
    strat_ok = USE_STRAT_CV and (GROUP_LABEL in df_ml.columns) and (df_ml[GROUP_LABEL].notna().any())

    cv_obj = _build_cv_object(
        n_splits=CV_N_SPLITS,
        n_repeats=CV_N_REPEATS,
        random_state=CV_RANDOM_SEED,
        stratified=strat_ok,
    )

    # Build a cv object that julearn expects (classifier=False for regression)
    cv_outer = check_cv(cv_obj, classifier=(PROBLEM_TYPE == "classification"))
    try:
        return _compute_cvmdsum(cv_outer)
    except Exception as e:
        warnings.warn(f"[CV mdsum] _compute_cvmdsum failed: {e}")
        return None

# %%
# ------------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------------
def scores_path(base_dir: str, model_key: str, target: str, feature: str) -> str:
    return os.path.join(base_dir, model_key, target, feature, 'scores.csv')

def read_scores_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for junk in ('Unnamed: 0', 'index'):
        if junk in df.columns:
            df = df.drop(columns=[junk])
    return df

def load_feature_frames_for(model_key: str, target: str, features: list[str]) -> dict[str, pd.DataFrame]:
    """Return dict feature_key -> df; skip features whose CSV is missing."""
    frames = {}
    for feat in features:
        fpath = scores_path(BASE_DIR, model_key, target, feat)
        try:
            df = read_scores_csv(fpath).copy()
            df['model']   = MODEL_LABEL.get(model_key, model_key)
            df['feature'] = FEATURE_LABEL.get(feat, feat)
            frames[feat]  = df
            print(f"[LOAD] {model_key:10s} | {target:6s} | {feat:24s} -> rows={len(df):4d} | {fpath}")
        except FileNotFoundError:
            warnings.warn(f"[SKIP] Missing: {fpath}")
    return frames

def concat_features(frames_by_feature: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames_by_feature:
        return pd.DataFrame()
    return pd.concat(list(frames_by_feature.values()), axis=0, ignore_index=True)

# %%
def plot_box_swarm_by_feature(df: pd.DataFrame,
                              model_label: str,
                              target: str,
                              metric: str,
                              order_features: list[str],
                              ylim=None,
                              save_path: str | None = None,
                              *,
                              fig_size: tuple[float, float] = (6, 6),   # <— edit per call
                              base_fig_in: float = 6.0,                 # scaling baseline
                              dpi: int = 300):
    """Boxplot across feature combinations for a given model+target, with size-aware styling."""
    if df.empty:
        print(f"[WARN] No data to plot for {model_label} | {target}")
        return
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in columns: {list(df.columns)}")

    # ----- fixed font sizes (pt) -----
    FS_TICK  = 10
    FS_LABEL = 12
    FS_TITLE = 14
    # ---------------------------------

    # Keep only necessary cols; drop NA metric rows
    d = df[['feature', metric]].dropna(subset=[metric]).copy()

    # Order axis using pretty labels
    order_labels = [FEATURE_LABEL.get(f, f) for f in order_features]
    d['feature'] = pd.Categorical(d['feature'], categories=order_labels, ordered=True)

    # --- scale visual elements w.r.t. figure inches (fonts stay fixed) ---
    w_in, h_in = fig_size
    scale = max(0.25, min(w_in, h_in) / base_fig_in)  # guard against tiny/huge figures

    # Baselines from your current style
    base_box_width = 0.60
    base_swarm_sz  = 4.0
    base_lw        = 1.0     # generic lines
    base_median_lw = 1.6

    box_width   = min(0.85, max(0.35, base_box_width * scale))
    swarm_size  = max(1.5,  base_swarm_sz * scale)    # swarm 'size' is in points (linear)
    lw          = max(0.6,  base_lw * scale)
    median_lw   = max(1.0,  base_median_lw * scale)

    fig, ax = plt.subplots(figsize=fig_size)

    # Outline-only boxplot
    sns.boxplot(
        data=d, x='feature', y=metric, ax=ax,
        order=order_labels, width=box_width, showfliers=False,
        boxprops=dict(facecolor='none', linewidth=lw, edgecolor='black'),
        medianprops=dict(color='black', linewidth=median_lw),
        whiskerprops=dict(color='black', linewidth=lw),
        capprops=dict(color='black', linewidth=lw),
    )

    # Swarm overlay with model color
    sns.swarmplot(
        data=d, x='feature', y=metric, ax=ax,
        order=order_labels,
        dodge=False, size=swarm_size, edgecolor='none',
        color=SWARM_COLORS.get(model_label, '#333333')
    )

    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # Apply fonts (unchanged)
    ax.set_title(f"{target} | {model_label}", fontsize=FS_TITLE)
    ax.set_xlabel('Feature Combination', fontsize=FS_LABEL)
    ax.set_ylabel(Y_LABELS.get(metric, metric), fontsize=FS_LABEL)
    ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
    ax.tick_params(axis='x', labelrotation=45)

    if ylim is not None:
        ax.set_ylim(*ylim)

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # save svg also
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, bbox_inches='tight')
        print(f"[SAVE] {svg_path}")
        print(f"[SAVE] {save_path}")

    plt.show()

# %%
# ------------------------------------------------------------------
# Stats helpers (mirrors cross-model script)
# ------------------------------------------------------------------
def infer_n_train_test(frames_nested: dict, n_train: int | None, n_test: int | None) -> tuple[int, int]:
    def _first_val(df, col):
        try:
            return int(df[col].iloc[0])
        except Exception:
            return None
    infer_train, infer_test = None, None
    for _feat, df in frames_nested.items():
        if infer_train is None and 'n_train' in df.columns:
            infer_train = _first_val(df, 'n_train')
        if infer_test is None and 'n_test' in df.columns:
            infer_test = _first_val(df, 'n_test')
        if infer_train is not None and infer_test is not None:
            break

    n_train_final = n_train if n_train is not None else infer_train
    n_test_final  = n_test  if n_test  is not None else infer_test

    # Fallback: rebuild sizes from raw data using your CV config
    if n_train_final is None or n_test_final is None:
        rebuilt = compute_cv_sizes_from_raw()
        if rebuilt is not None:
            n_train_final, n_test_final = rebuilt

    if n_train_final is None or n_test_final is None:
        raise ValueError(
            "n_train/n_test are required by corrected_ttest. "
            "Set N_TRAIN and N_TEST above, ensure CSVs contain 'n_train'/'n_test', "
            "or provide RAW_DATA_* so they can be reconstructed."
        )
    return int(n_train_final), int(n_test_final)

def extract_cv_mdsum(frames_by_feature: dict) -> str | None:
    # 1) Try to take the first non-null cv_mdsum from the loaded CSVs (as STRING)
    for _f, df in frames_by_feature.items():
        if 'cv_mdsum' in df.columns and not df['cv_mdsum'].isna().all():
            try:
                val = df['cv_mdsum'].iloc[0]
                if pd.notna(val):
                    return str(val)
            except Exception:
                continue

    # 2) Fallback: compute cv_mdsum from raw data using the same CV config
    cvm = compute_cv_mdsum_from_raw()
    if cvm is not None:
        return str(cvm)

    # 3) Nothing available
    warnings.warn("[CV mdsum] Could not infer or compute cv_mdsum; corrected_ttest may fail.")
    return None

def standardize_df(df: pd.DataFrame, n_train: int, n_test: int, cv_mdsum_value, k_folds: int) -> pd.DataFrame:
    out = df.copy()
    out['n_train'] = n_train
    out['n_test']  = n_test

    # cv_mdsum is a STRING id; only set if missing/NaN
    if 'cv_mdsum' not in out.columns or out['cv_mdsum'].isna().all():
        if cv_mdsum_value is not None:
            out['cv_mdsum'] = cv_mdsum_value

    for col in TRAIN_METRICS:
        if col not in out.columns:
            out[col] = 0.0

    # Flatten (repeat, fold) -> unique fold index expected by corrected_ttest
    if 'fold' in out.columns and 'repeat' in out.columns:
        out['fold'] = out['fold'] + (k_folds * (out['repeat'] - 1)) - 1
        out['repeat'] = 0
    return out

def run_corrected_ttests(prepared_frames_ordered: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_df = corrected_ttest(*prepared_frames_ordered)
    p = stats_df['p-val-corrected'].values
    stats_df['significance'] = np.select(
        [
            p <= 1e-4,
            (p > 1e-4) & (p <= 1e-3),
            (p > 1e-3) & (p <= 1e-2),
            (p > 1e-2) & (p <= 0.05),
        ],
        ['****', '***', '**', '*'],
        default='ns'
    )
    focus = stats_df[stats_df['metric'].isin(METRICS_OF_INTEREST)].copy()
    focus['significant'] = focus['p-val-corrected'] < 0.05
    return stats_df, focus

# %%
# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_DIR = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/model_results'
FEATURES = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov', 'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle']
# FEATURES = ['Sleep', 'Sleep_APOE', 'Sleep_APOE_Shuffle',
#             'Sleep_Cov', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle',
#             'Sleep_Cov_Brain', 'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle']
TARGETS  = ['Stroop', 'Memory']
MODELS   = ['AutoGluon']  # 'XGBoost', 'AutoGluon'

# Plot controls
PLOT_METRIC = 'test_spearmanr'      # or 'test_r_corr' or 'test_r2' or 'test_spearmanr'
YLIM        = (-0.05, 0.60)     # e.g., (-0.15, 0.60) or None for auto
SAVE_PLOTS  = True
OUT_DIR     = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs'

SAVE_STATS = True
STATS_DIR = os.path.join(OUT_DIR, "stats_tables")
if SAVE_STATS:
    os.makedirs(STATS_DIR, exist_ok=True)

def _safe_name(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)

# Pretty labels
MODEL_LABEL = {'XGBoost': 'XGBoost', 'AutoGluon': 'AutoGluon'}
FEATURE_LABEL = {
    'Sleep':                       'Sleep',
    'Cov':                         'Demographics',
    'Sleep_Cov':                   'Sleep, Demo',
    'Sleep_Shuffle_Cov':           'Sleep (Shuffle), Demo',
    'Brain':                       'Brain',
    'Sleep_Cov_Brain':             'Sleep, Demo, Brain',
    'Sleep_Cov_Brain_Shuffle':     'Sleep, Demo, Brain (Shuffle)',
    'Sleep_APOE':                  'Sleep, APOE',
    'Sleep_APOE_Shuffle':          'Sleep, APOE (Shuffle)',
    'Sleep_Cov_APOE':              'Sleep, Demo, APOE',
    'Sleep_Cov_APOE_Shuffle':      'Sleep, Demo, APOE (Shuffle)',
    'Sleep_Cov_Brain_APOE':        'Sleep, Demo, Brain, APOE',
    'Sleep_Cov_Brain_APOE_Shuffle':'Sleep, Demo, Brain, APOE (Shuffle)',
}
Y_LABELS = {'test_r_corr': 'Pearson Correlation Coefficient', 'test_r2': 'R²', 'test_spearmanr': "Spearman Correlation coefficient"}  # Spearman's rank correlation coefficient

K_FOLDS = 5
# Training/test sizes for corrected_ttest (set both or infer from CSVs if present)
N_TRAIN = None
N_TEST  = None

# Raw dataset config (used only if we need to rebuild CV sizes)
RAW_DATA_DIR  = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
RAW_DATA_CSV  = 'SHIP_Trend_dataset_renamed.csv'
GROUP_LABEL   = 'Group_Age_SEX'     # column used for stratification if available
USE_STRAT_CV  = True                # try stratified CV if GROUP_LABEL is present
CV_N_SPLITS   = 5
CV_N_REPEATS  = 10
CV_RANDOM_SEED = 42

# Problem type for check_cv / cv_mdsum
PROBLEM_TYPE = "regression"

TRAIN_METRICS = [
    'train_r2',
    'train_neg_mean_absolute_error',
    'train_neg_root_mean_squared_error',
    'train_r_corr',
    'train_spearmanr',
]
METRICS_OF_INTEREST = ['test_r_corr', 'test_r2', 'test_spearmanr']


# Optional fixed palette per (pretty) feature label (used by swarm)
FEATURE_COLORS = {
    'Sleep':                              '#5DA5DA',
    'Demographics':                       '#60BD68',
    'Sleep, Demo':                        '#F17CB0',
    'Sleep (Shuffle), Demo':              '#B2912F',
    'Brain':                              '#B276B2',
    'Sleep, Demo, Brain':                 '#F15854',
    'Sleep, Demo, Brain (Shuffle)':       '#4D4D4D',
}

SWARM_COLORS = {
    'XGBoost':   '#FD7F5B',
    'AutoGluon': '#FD4531',
}

# %%
# ------------------------------------------------------------------
# Execute interactively
#   - Loops over targets and the two models
#   - Plots a figure per (model, target) comparing FEATURES
# ------------------------------------------------------------------
# Execute interactively (+ stats)
if SAVE_PLOTS:
    os.makedirs(OUT_DIR, exist_ok=True)

stats = {}
for target in TARGETS:
    for model_key in MODELS:
        frames = load_feature_frames_for(model_key, target, FEATURES)
        if not frames:
            print(f"[WARN] No data for {model_key} | {target} — skipping.")
            continue

        # Infer sample sizes and prepare per-feature frames
        n_train, n_test = infer_n_train_test(frames, N_TRAIN, N_TEST)
        cv_mdsum_val = extract_cv_mdsum(frames)

        prepared = []
        prepared_stats = []  # for stats (model column becomes feature label)

        for feat in FEATURES:
            if feat not in frames:
                continue
            df_std = standardize_df(frames[feat], n_train, n_test, cv_mdsum_val, K_FOLDS)
            frames[feat] = df_std
            prepared.append(df_std)

            # Clone for stats and set "model" to the FEATURE label so corrected_ttest prints them
            df_stats = df_std.copy()
            df_stats['model'] = FEATURE_LABEL.get(feat, feat)
            prepared_stats.append(df_stats)

        # # Run corrected t-tests across FEATURES for this model+target
        # stats_full, stats_focus = run_corrected_ttests(prepared)
        # stats.setdefault(target, {})[model_key] = {'full': stats_full, 'focus': stats_focus}

        # Run corrected t-tests across FEATURES for this model+target
        stats_full, stats_focus = run_corrected_ttests(prepared_stats)

        # Rename for readability: model_1/model_2 -> feature_1/feature_2
        stats_full_disp = stats_full.rename(columns={'model_1': 'feature_1', 'model_2': 'feature_2'})
        stats_focus_disp = stats_focus.rename(columns={'model_1': 'feature_1', 'model_2': 'feature_2'})

        # Store the renamed versions
        stats.setdefault(target, {})[model_key] = {'full': stats_full_disp, 'focus': stats_focus_disp}

        # Console view
        # print(f"\n=== Corrected t-test (FULL) : {target} | {model_key} ===")
        # print(stats_full.head(50).to_string(index=False))
        # print(f"\n=== Corrected t-test (FOCUS: {', '.join(METRICS_OF_INTEREST)}) : {target} | {model_key} ===")
        # print(stats_focus.to_string(index=False))
        # print(f"\n=== Corrected t-test (FULL) : {target} | {model_key} ===")
        # print(stats_full_disp.head(50).to_string(index=False))
        print(f"\n=== Corrected t-test (FOCUS: {', '.join(METRICS_OF_INTEREST)}) : {target} | {model_key} ===")
        print(stats_focus_disp.to_string(index=False))

        # build a consistent tag for filenames
        metric_tag = 'corr' if PLOT_METRIC == 'test_r_corr' else PLOT_METRIC
        model_tag = MODEL_LABEL.get(model_key, model_key)

        if SAVE_STATS:
            base = _safe_name(f"{target}_{model_tag}_{metric_tag}")

            # CSV outputs
            f_full_csv = os.path.join(STATS_DIR, f"{base}_stats_full.csv")
            f_focus_csv = os.path.join(STATS_DIR, f"{base}_stats_focus.csv")
            stats_full_disp.to_csv(f_full_csv, index=False)
            stats_focus_disp.to_csv(f_focus_csv, index=False)

            print(f"[SAVE] {f_full_csv}")
            print(f"[SAVE] {f_focus_csv}")

        # Plot
        df_all = concat_features(frames)
        if df_all.empty:
            print(f"[WARN] No data for {model_key} | {target} — skipping plot.")
            continue

        save_path = None
        if SAVE_PLOTS:
            metric_tag = 'corr' if PLOT_METRIC == 'test_r_corr' else PLOT_METRIC
            fname = f"{target}_{MODEL_LABEL.get(model_key, model_key)}_{metric_tag}.png"  # add APOE for APOE models comparison
            fname = fname.replace(' ', '_').replace('+', 'plus')
            save_path = os.path.join(OUT_DIR, fname)

        plot_box_swarm_by_feature(
            df=df_all,
            model_label=MODEL_LABEL.get(model_key, model_key),
            target=target,
            metric=PLOT_METRIC,
            order_features=FEATURES,
            ylim=YLIM,
            save_path=save_path,
            fig_size=(5, 5),
        )

# Handy convenience vars for the last (target, model) if you want them in REPL
if 'stats' in locals():
    last_target = TARGETS[-1]
    last_model  = MODELS[-1]
    stats_full_last  = stats[last_target][last_model]['full']
    stats_focus_last = stats[last_target][last_model]['focus']
