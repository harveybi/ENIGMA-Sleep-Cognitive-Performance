import os

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
    # Drop common stray index columns
    for junk in ('Unnamed: 0', 'index'):
        if junk in df.columns:
            df = df.drop(columns=[junk])
    return df

def load_all_frames(base_dir: str, feature: str, targets: list[str], model_keys: list[str]) -> dict:
    frames = {}
    for target in targets:
        frames[target] = {}
        for mk in model_keys:
            fpath = scores_path(base_dir, mk, target, feature)
            df = read_scores_csv(fpath)
            frames[target][mk] = df
            print(f"[LOAD] {mk:11s} | {target:6s} | rows={len(df):4d} -> {fpath}")
    return frames

# %%
# ------------------------------------------------------------------
# Standardization helpers
# ------------------------------------------------------------------
def infer_n_train_test(frames_nested: dict, n_train: int | None, n_test: int | None) -> tuple[int, int]:
    def _first_val(df, col):
        try:
            return int(df[col].iloc[0])
        except Exception:
            return None
    infer_train, infer_test = None, None
    for _tgt, frames_by_model in frames_nested.items():
        for _mk, df in frames_by_model.items():
            if infer_train is None and 'n_train' in df.columns:
                infer_train = _first_val(df, 'n_train')
            if infer_test is None and 'n_test' in df.columns:
                infer_test = _first_val(df, 'n_test')
            if infer_train is not None and infer_test is not None:
                break
    n_train_final = n_train if n_train is not None else infer_train
    n_test_final  = n_test  if n_test  is not None else infer_test
    if n_train_final is None or n_test_final is None:
        raise ValueError(
            "n_train/n_test are required by corrected_ttest. "
            "Set N_TRAIN and N_TEST above, or ensure CSVs contain 'n_train' and 'n_test'."
        )
    return int(n_train_final), int(n_test_final)

def extract_cv_mdsum(frames_by_model: dict) -> float | None:
    for _k, df in frames_by_model.items():
        if 'cv_mdsum' in df.columns and not df['cv_mdsum'].isna().all():
            try:
                return df['cv_mdsum'].iloc[0]
            except Exception:
                continue
    return None

def standardize_df(df: pd.DataFrame, model_key: str, n_train: int, n_test: int,
                   cv_mdsum_value, k_folds: int) -> pd.DataFrame:
    out = df.copy()
    out['n_train'] = n_train
    out['n_test'] = n_test
    out['model'] = MODEL_LABEL.get(model_key, model_key)
    if 'cv_mdsum' not in out.columns or out['cv_mdsum'].isna().all():
        if cv_mdsum_value is not None:
            out['cv_mdsum'] = cv_mdsum_value
    for col in TRAIN_METRICS:
        if col not in out.columns:
            out[col] = 0.0
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

def concat_with_model_key(frames_by_model: dict) -> pd.DataFrame:
    cat = []
    for mk, df in frames_by_model.items():
        d = df.copy()
        if 'model' not in d.columns:
            d['model'] = MODEL_LABEL.get(mk, mk)
        cat.append(d)
    return pd.concat(cat, axis=0, ignore_index=True)

def _safe(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)

# %%
# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_DIR = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/model_results'
FEATURE = 'Sleep_Cov_Brain'  # 'Sleep', 'Cov', 'Sleep_Cov', 'Brain', 'Sleep_Cov_Brain', 'Sleep_Shuffle_Cov'
TARGETS = ['Stroop', 'Memory']
MODELS = ['Linear', 'Ridge', 'SVM-linear', 'SVM-rbf', 'rf', 'XGBoost', 'AutoGluon']  # display/analysis order
K_FOLDS = 5  # used to flatten (repeat, fold) → unique fold index

# Training/test sizes for corrected_ttest (set both or infer from CSVs if present)
N_TRAIN = None  # put an int or leave None to infer from CSV columns
N_TEST = None  # put an int or leave None to infer from CSV columns

# Optional plotting
MAKE_PLOTS = True
PLOT_METRIC = 'test_spearmanr'      # or 'test_r_corr' or 'test_r2' or 'test_spearmanr'

# Columns / metrics
MODEL_LABEL = {
    'Linear': 'Linear Regression',
    'Ridge': 'Ridge Regression',
    'SVM-linear': 'SVM-linear',
    'SVM-rbf': 'SVM-rbf',
    'rf': 'Random Forest',
    'XGBoost': 'XGBoost',
    'AutoGluon': 'AutoGluon',
}
TRAIN_METRICS = [
    'train_r2',
    'train_neg_mean_absolute_error',
    'train_neg_root_mean_squared_error',
    'train_r_corr',
    'train_spearmanr',
]
METRICS_OF_INTEREST = ['test_r_corr', 'test_r2', 'test_spearmanr']

FEATURE_LABEL = {
    'Sleep': 'Sleep',
    'Cov': 'Demographics',
    'Sleep_Cov': 'Sleep, Demo',
    'Brain': 'Brain',
    'Sleep_Cov_Brain': 'Sleep, Demo, Brain',
    'Sleep_Shuffle_Cov': 'Sleep (Shuffle) + Demo',
}

# %%
# ------------------------------------------------------------------
# Execute (interactive-friendly)
# ------------------------------------------------------------------
# 1) Load
frames = load_all_frames(BASE_DIR, FEATURE, TARGETS, MODELS)

# 2) Determine n_train / n_test
n_train, n_test = infer_n_train_test(frames, N_TRAIN, N_TEST)
print(f"\n[INFO] Using n_train={n_train}, n_test={n_test}")

# 3) Standardize per target and run corrected t-tests
stats = {}
for target in TARGETS:
    frames_by_model = frames[target]
    cv_mdsum_val = extract_cv_mdsum(frames_by_model)
    prepared = []
    for mk in MODELS:
        df_std = standardize_df(frames_by_model[mk], mk, n_train, n_test, cv_mdsum_val, K_FOLDS)
        frames_by_model[mk] = df_std
        prepared.append(df_std)
    stats_full, stats_focus = run_corrected_ttests(prepared)
    stats[target] = {'full': stats_full, 'focus': stats_focus}

# 4) Inspect results in console
print("\n=== Corrected t-test (FULL) : Stroop ===")
print(stats['Stroop']['full'].head(50).to_string(index=False))
print("\n=== Corrected t-test (FOCUS: test_r_corr & test_r2) : Stroop ===")
print(stats['Stroop']['focus'].to_string(index=False))

print("\n=== Corrected t-test (FULL) : Memory ===")
print(stats['Memory']['full'].head(50).to_string(index=False))
print("\n=== Corrected t-test (FOCUS: test_r_corr & test_r2) : Memory ===")
print(stats['Memory']['focus'].to_string(index=False))

# %%
# 5) Optional quick barplots for the chosen metric
if MAKE_PLOTS:
    '''
    Swarmplot dot color:
    Linear Regression: B9D5E0
    Ridge Regression: 92C0D5
    SVM-linear: 88BDF3
    SVM-rbf: FFDBBF
    Random Forest: FEAA95
    XGBoost: FD7F5B
    AutoGluon: FD4531
    '''
    SWARM_COLORS = {
        'Linear Regression': '#B9D5E0',
        'Ridge Regression': '#92C0D5',
        'SVM-linear': '#88BDF3',
        'SVM-rbf': '#FFDBBF',
        'Random Forest': '#FEAA95',
        'XGBoost': '#FD7F5B',
        'AutoGluon': '#FD4531',
    }

    # Style: remove top/right spines
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Order of models on x-axis (human-readable labels)
    order_labels = [MODEL_LABEL.get(mk, mk) for mk in MODELS]

    # Optional fixed y-limits; set to None to auto-scale
    YLIM = None  # e.g., YLIM = (-0.15, 0.60)

    # Pretty y-axis labels by metric
    Y_LABELS = {
        'test_r_corr': 'Pearson Correlation Coefficient',
        'test_r2': 'R²',
        'test_spearmanr': "Spearman correlation coefficient",
    }

    feature_title = FEATURE_LABEL.get(FEATURE, FEATURE)
    metric_title = Y_LABELS.get(PLOT_METRIC, PLOT_METRIC)

    # ----- fixed font sizes (pt) -----
    FS_TICK = 10
    FS_LABEL = 12
    FS_TITLE = 14
    # ---------------------------------

    for target in TARGETS:
        df_concat = concat_with_model_key(frames[target])  # has 'model' as labels already

        fig, ax = plt.subplots(figsize=(6, 6))

        # Boxplot (no fill)
        sns.boxplot(
            data=df_concat,
            x='model', y=PLOT_METRIC,
            order=order_labels, ax=ax,
            width=0.6,
            showfliers=False,
            boxprops=dict(facecolor='none'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
        )

        # Swarm overlay
        sns.swarmplot(
            data=df_concat, x='model', y=PLOT_METRIC,
            order=order_labels, hue='model', palette=SWARM_COLORS,
            dodge=False, ax=ax, size=4, edgecolor='none'
        )

        # Labels / title
        # ax.set_title(f'{target} | {FEATURE} | {PLOT_METRIC}')
        ax.set_title(f'{target} | {feature_title}', fontsize=FS_TITLE)
        ax.set_xlabel('Model', fontsize=FS_LABEL)
        ax.set_ylabel(Y_LABELS.get(PLOT_METRIC, PLOT_METRIC), fontsize=FS_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=FS_TICK)

        # Optional fixed y-limits
        if YLIM is not None:
            ax.set_ylim(*YLIM)

        # Rotate x tick labels
        ax.tick_params(axis='x', rotation=45)

        # Final touches
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.show()

# 6) Handy filtered views you can poke at interactively:
stats_stroop_df_full  = stats['Stroop']['full']
stats_memory_df_full  = stats['Memory']['full']
stats_stroop_df_focus = stats['Stroop']['focus']
stats_memory_df_focus = stats['Memory']['focus']

# If you want 'test_r_corr' only:
stats_stroop_df_corr  = stats_stroop_df_full.query("metric == 'test_r_corr'").copy()
stats_memory_df_corr  = stats_memory_df_full.query("metric == 'test_r_corr'").copy()

# %%
# FEATURES_TO_PLOT = ['Cov', 'Sleep', 'Sleep_Cov', 'Brain', 'Sleep_Cov_Brain', 'Sleep_Shuffle_Cov']
FEATURES_TO_PLOT = ['Cov', 'Sleep_Cov', 'Sleep_Cov_Brain']
SAVE_PLOTS = True  # set True to write PNGs
OUT_DIR = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Results_making/stats_figs/new_plot_cross_model'  # used if SAVE_PLOTS=True

SAVE_STATS = True
STATS_DIR = os.path.join(OUT_DIR, "tables")  # or any path you want
os.makedirs(STATS_DIR, exist_ok=True)

# Pretty y-axis labels by metric
Y_LABELS = {'test_r_corr': 'Pearson Correlation Coefficient', 'test_r2': 'R²', 'test_spearmanr': "Spearman correlation coefficient"}

# Swarm colors (human-readable model labels)
SWARM_COLORS = {
    'Linear Regression': '#B9D5E0',
    'Ridge Regression':  '#92C0D5',
    'SVM-linear':        '#88BDF3',
    'SVM-rbf':           '#FFDBBF',
    'Random Forest':     '#FEAA95',
    'XGBoost':           '#FD7F5B',
    'AutoGluon':         '#FD4531',
}

# Style: remove top/right spines once
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Optional fixed y-limits; set to None to auto-scale
YLIM = None  # e.g., (-0.15, 0.60)
YLIM_DEFAULT = None
YLIM_PER_TARGET = {
    'Stroop': (-0.05, 0.60),
    'Memory': (-0.10, 0.50),
}

# Store per-target → per-feature stats
stats_all = {}

if SAVE_PLOTS:
    os.makedirs(OUT_DIR, exist_ok=True)

# Keep model order once
order_labels = [MODEL_LABEL.get(mk, mk) for mk in MODELS]

for target in TARGETS:
    print(f"\n===== TARGET: {target} =====")
    stats_all[target] = {}

    # Per-target y-limit (fallback to global/default)
    yl = YLIM_PER_TARGET.get(target, YLIM_DEFAULT)

    for FEATURE in FEATURES_TO_PLOT:
        print(f"\n--- FEATURE: {FEATURE} ---")
        feature_title = FEATURE_LABEL.get(FEATURE, FEATURE)
        metric_title  = Y_LABELS.get(PLOT_METRIC, PLOT_METRIC)

        # 1) Load frames for this FEATURE but only this TARGET
        frames = load_all_frames(BASE_DIR, FEATURE, [target], MODELS)

        # 2) Determine n_train / n_test for this feature (using the same structure)
        n_train, n_test = infer_n_train_test(frames, N_TRAIN, N_TEST)
        print(f"[INFO] Using n_train={n_train}, n_test={n_test}")

        # 3) Standardize per model and run corrected t-tests (across MODELS)
        frames_by_model = frames[target]
        cv_mdsum_val = extract_cv_mdsum(frames_by_model)

        prepared = []
        for mk in MODELS:
            df_std = standardize_df(frames_by_model[mk], mk, n_train, n_test, cv_mdsum_val, K_FOLDS)
            frames_by_model[mk] = df_std
            prepared.append(df_std)

        stats_full, stats_focus = run_corrected_ttests(prepared)
        stats_all[target][FEATURE] = {'full': stats_full, 'focus': stats_focus}

        if SAVE_STATS:
            safe_feat = _safe(feature_title)
            safe_tgt = _safe(target)
            safe_met = _safe(PLOT_METRIC)

            f_full = os.path.join(STATS_DIR, f"{safe_tgt}_{safe_feat}_stats_full.csv")
            f_focus = os.path.join(STATS_DIR, f"{safe_tgt}_{safe_feat}_stats_focus.csv")

            stats_full.to_csv(f_full, index=False)
            stats_focus.to_csv(f_focus, index=False)
            print(f"[SAVE] {f_full}")
            print(f"[SAVE] {f_focus}")

        # (Optional) inspect briefly
        print(stats_focus.to_string(index=False))

        # 4) Plot (single figure per TARGET×FEATURE)
        if MAKE_PLOTS:
            df_concat = concat_with_model_key(frames_by_model)  # has 'model' labels

            # ----- fixed font sizes (pt) -----
            FS_TICK = 10
            FS_LABEL = 12
            FS_TITLE = 14
            # ---------------------------------

            # ---- size + scaling controls (edit FIG_SIZE only) ----
            FIG_SIZE = (4, 4)  # was (6, 6); shrink to make text appear larger
            BASE_FIG_IN = 6.0  # baseline inches for scaling
            w_in, h_in = FIG_SIZE
            scale = max(0.25, min(w_in, h_in) / BASE_FIG_IN)  # guard extremes

            # Visual baselines (from your current style)
            base_box_width = 0.60
            base_swarm_sz = 4.0
            base_lw = 1.0
            base_median_lw = 1.6

            # Scaled visuals (fonts remain fixed)
            box_width = min(0.85, max(0.35, base_box_width * scale))
            swarm_size = max(1.5, base_swarm_sz * scale)  # points (linear)
            lw = max(0.6, base_lw * scale)
            median_lw = max(1.0, base_median_lw * scale)
            # ------------------------------------------------------

            fig, ax = plt.subplots(figsize=FIG_SIZE)

            # Boxplot (outline only)
            sns.boxplot(
                data=df_concat,
                x='model', y=PLOT_METRIC,
                order=order_labels, ax=ax,
                width=box_width, showfliers=False,
                boxprops=dict(facecolor='none', linewidth=lw, edgecolor='black'),
                medianprops=dict(color='black', linewidth=median_lw),
                whiskerprops=dict(color='black', linewidth=lw),
                capprops=dict(color='black', linewidth=lw),
            )

            # Swarm overlay colored by model
            sns.swarmplot(
                data=df_concat,
                x='model', y=PLOT_METRIC,
                order=order_labels, hue='model', palette=SWARM_COLORS,
                dodge=False, ax=ax, size=swarm_size, edgecolor='none'
            )
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            # Labels / fonts (unchanged)
            ax.set_title(f'{target} | {feature_title}', fontsize=FS_TITLE)
            ax.set_xlabel('Model', fontsize=FS_LABEL)
            ax.set_ylabel(metric_title, fontsize=FS_LABEL)
            ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
            ax.tick_params(axis='x', rotation=45)

            if yl is not None:
                ax.set_ylim(*yl)

            sns.despine(ax=ax)
            plt.tight_layout()

            if SAVE_PLOTS:
                safe_feat = feature_title.replace(' ', '_').replace('+', 'plus')
                fname = f"{target}_{safe_feat}_{PLOT_METRIC}.png"
                fpath = os.path.join(OUT_DIR, fname)
                plt.savefig(fpath, dpi=300, bbox_inches='tight')
                # save svg also
                fpath_svg = fpath.replace('.png', '.svg')
                plt.savefig(fpath_svg, bbox_inches='tight')
                print(f"[SAVE] {fpath}")

            plt.show()

# Handy convenience vars (last target/feature)
if TARGETS and FEATURES_TO_PLOT:
    last_target = TARGETS[-1]
    last_feat   = FEATURES_TO_PLOT[-1]
    stats_df_full  = stats_all[last_target][last_feat]['full']
    stats_df_focus = stats_all[last_target][last_feat]['focus']
