import os
import sys
import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import contextlib

import numpy as np
import pandas as pd

import shap
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Style
# -------------------------------
sns.set_context("paper")
mpl.rcParams['font.family']      = 'sans-serif'
mpl.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})


import warnings
# Ignore FutureWarning in the whole script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# -------------------------------
# Config
# -------------------------------
# load the feature lists from feature_lists.pkl
feature_lists_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/feature_lists.pkl'  # where the feature lists are saved
with open(feature_lists_path, 'rb') as f:
    feature_lists = pickle.load(f)

Sleep = feature_lists[0]
Cov = feature_lists[1]
APOE4 = feature_lists[2]
TIV = feature_lists[3]
Thickness_DK = feature_lists[4]
Thickness_Schaefer = feature_lists[5]
Area_DK = feature_lists[6]
Area_Schaefer = feature_lists[7]
Subcortical = feature_lists[8]

MODEL_DIR = Path("/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models")
MODEL  = "AutoGluon"  # 'AutoGluon' or 'XGBoost'
TARGET = ("Memory")  # 'Stroop' or 'Memory' (set as needed)
FEATURE_COMB = "Sleep_Cov_Brain"  # 'Sleep_Cov' or 'Sleep_Cov_Brain'
if FEATURE_COMB == "Sleep_Cov_Brain":
    MODEL = "AutoGluon_no_DK"

# Cohorts per target
COHORTS_BY_TARGET = {
    "Stroop": ['train', 'VETSA', 'Pitts', 'Liege'],
    "Memory": ['train', 'VETSA', 'Liege', 'KI', 'Pitts', 'Juelich'],
}
COHORTS = COHORTS_BY_TARGET[TARGET]

SITE_LABEL = {
    'train': 'Greifswald',
    'VETSA': 'San Diego',
    'Liege': 'Liège',
    'KI': 'Stockholm',
    'Pitts': 'Pittsburgh',
    'Juelich': 'Jülich',
}

LABEL_REPLACEMENTS = {
    "Age_at_Scan": "Age",
    'Self_Sleep_Dur': 'Self Sleep Duration',
    'Self_Sleep_Eff': 'Self Sleep Efficiency',
    'PSG_Sleep_Dur': 'PSG Sleep Duration',
    'PSG_Sleep_Eff': 'PSG Sleep Efficiency',
    "Depression_score": "Depressive score",
}

TARGETS_BY_SITE = {
    'VETSA':   ['Stroop', 'Memory_Digit', 'Memory_Letter', 'Stroop_rgo_age',
                'Memory_Digit_rgo_age', 'Memory_Letter_rgo_age'],
    'EMC':     ['Stroop', 'Stroop_rgo_age'],
    'Liege':   ['Stroop', 'Memory', 'Stroop_rgo_age', 'Memory_rgo_age'],
    'KI':      ['Memory', 'Memory_rgo_age'],
    'Pitts':   ['Executive', 'Memory_Letter', 'Memory_Spatial', 'Executive_rgo_age',
                'Memory_Letter_rgo_age', 'Memory_Spatial_rgo_age'],
    'Juelich': ['Memory_Letter', 'Memory_Spatial', 'Memory_Letter_rgo_age', 'Memory_Spatial_rgo_age'],
}

# Feature ordering mode: 'train' | 'cohort' | 'none'
ORDER_MODE = 'none'

OUT_DIR = MODEL_DIR / MODEL / TARGET / FEATURE_COMB / "SHAP" / "plots_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Build canonical group sets ---
SLEEP_SET = set(Sleep)
COV_SET = set(Cov)
THICKNESS_SET = set(Thickness_DK) | set(Thickness_Schaefer)
AREA_SET = set(Area_DK) | set(Area_Schaefer)
SUBCORT_SET = set(Subcortical)

GROUP_SETS = {
    "Sleep": SLEEP_SET,
    "Cov": COV_SET,
    "Thickness": THICKNESS_SET,
    "Area": AREA_SET,
    "Subcortical": SUBCORT_SET,
}

BRAIN_GROUPS = {"Thickness", "Area", "Subcortical"}
GROUP_ORDER  = ["Sleep", "Cov", "Thickness", "Area", "Subcortical"]

# Deterministic name->group map
NAME_TO_GROUP: dict[str, str] = {}
for g in GROUP_ORDER:
    for name in GROUP_SETS[g]:
        NAME_TO_GROUP.setdefault(name, g)

# --- Feature count reduction (no L/R collapse) ---
LR_COLLAPSE = False
SELECTION_MODE = 'cohort'
CUM_THRESHOLD = 0.85
MIN_PER_GROUP = {'Thickness':10, 'Area':10, 'Subcortical':10}
MAX_PER_GROUP = {'Thickness':10, 'Area':10, 'Subcortical':10}

# Plot settings
SAVE_PNG = False  # True or False
SAVE_SVG = False  # True or False
SHOW_FIG = True  # True or False
DPI = 600  # 300

# %%
# -------------------------------
# Utils
# -------------------------------
def shap_dir_for(model: str, target: str, feature_comb: str) -> Path:
    return MODEL_DIR / model / target / feature_comb / "SHAP"

def shap_pickle_path(cohort: str) -> Path:
    base = shap_dir_for(MODEL, TARGET, FEATURE_COMB)
    if cohort == "train":
        return base / "explanation_train_set_train.pkl"
    return base / f"explanation_train_set_test_{cohort}.pkl"

def load_explanation(pkl_path: Path) -> shap.Explanation:
    with open(pkl_path, "rb") as f:
        exp = pickle.load(f)
    if not isinstance(exp, shap.Explanation):
        raise TypeError(f"Pickle at {pkl_path} is not a shap.Explanation (got {type(exp)})")
    return exp

def safe_feature_names(names: List[str]) -> List[str]:
    return [LABEL_REPLACEMENTS.get(n, n) for n in names]

def mean_abs_shap_by_feature(exp: shap.Explanation) -> pd.Series:
    vals = np.asarray(exp.values)
    if vals.ndim != 2:
        vals = np.abs(vals).sum(axis=-1)
    mas = np.abs(vals).mean(axis=0)
    return pd.Series(mas, index=list(exp.feature_names))

def order_from_train(train_exp: shap.Explanation) -> List[str]:
    s = mean_abs_shap_by_feature(train_exp).sort_values(ascending=False)
    return list(s.index)

def ensure_order_subset(order: List[str], names: List[str]) -> List[str]:
    avail = set(names)
    return [n for n in order if n in avail]

def reindex_explanation(exp: shap.Explanation,
                        feature_order: Optional[List[str]]) -> shap.Explanation:
    if not feature_order:
        return exp
    idx_map = {n: i for i, n in enumerate(exp.feature_names)}
    cols = [idx_map[n] for n in feature_order if n in idx_map]
    if not cols or len(cols) == len(exp.feature_names) and all(i == cols[i] for i in range(len(cols))):
        return exp
    new_vals = np.asarray(exp.values)[:, cols]
    new_data = None
    if exp.data is not None and hasattr(exp.data, "shape"):
        try:
            new_data = np.asarray(exp.data)[:, cols]
        except Exception:
            new_data = exp.data
    return shap.Explanation(
        values=new_vals, base_values=exp.base_values,
        data=new_data, feature_names=[exp.feature_names[i] for i in cols]
    )

def pretty_target_label(t: str) -> str:
    if t.endswith('_rgo_age'):
        base = t[:-8].replace('_', ' ')
        return f"{base} (regress out age)"
    return t.replace('_', ' ')

def concrete_target_for_site(site: str, target_type: str) -> str:
    if site == 'train':
        return target_type
    options = TARGETS_BY_SITE.get(site, [])
    if target_type == 'Stroop':
        prefix = 'Executive' if site == 'Pitts' else 'Stroop'
        choices = [t for t in options if t.startswith(prefix) and not t.endswith('_rgo_age')]
    else:
        choices = [t for t in options if ('Memory' in t) and not t.endswith('_rgo_age')]
    return choices[0] if choices else target_type

def feature_group(name: str) -> str:
    return NAME_TO_GROUP.get(name, "Cov")

def report_unclassified(exp: shap.Explanation) -> None:
    names = set(exp.feature_names)
    known  = set(NAME_TO_GROUP.keys())
    unknown = sorted(list(names - known))
    if unknown:
        print(f"[warn] {len(unknown)} feature(s) not in any list (defaulting to 'Cov'):")
        for u in unknown[:15]:
            print("   -", u)
        if len(unknown) > 15:
            print("   ...")

def simple_beeswarm(y, nbins=None, width=1.0):
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6
    nbins = int(nbins)
    x = np.zeros(len(y))
    ylo = np.min(y); yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)
    i = np.arange(len(y))
    ibs, ybs = [0]*nbins, [0]*nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))
    dx = width / max(1, (nmax // 2))
    for i_bin, y_bin in zip(ibs, ybs):
        if len(i_bin) > 1:
            j = len(i_bin) % 2
            i_sorted = i_bin[np.argsort(y_bin)]
            a = i_sorted[j::2]; b = i_sorted[j+1::2]
            x[a] = (0.5 + j/3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j/3 + np.arange(len(b))) * -dx
    return x

def long_df_from_explanation(exp: shap.Explanation) -> pd.DataFrame:
    vals = np.asarray(exp.values)
    n, p = vals.shape
    names = list(exp.feature_names)
    X = np.asarray(exp.data) if (exp.data is not None and hasattr(exp.data, "shape")) else None
    recs = []
    for j, fname in enumerate(names):
        g = feature_group(fname)
        sh_j = vals[:, j]
        feat_j = X[:, j] if X is not None else np.full(n, np.nan)
        recs.append(pd.DataFrame({
            "name": fname, "group": g,
            "shap": sh_j, "feature": feat_j,
            "sample_id": np.arange(n, dtype=int) + 1
        }))
    df = pd.concat(recs, axis=0, ignore_index=True)
    return df

# %%
# -------------------------------
# Plot: beeswarm + top-importance bar
# -------------------------------
def plot_combined(exp: shap.Explanation,
                  title: str,
                  out_stem: Path,
                  feature_order: Optional[List[str]],
                  fig_size: Optional[Tuple[float, float]] = None,
                  max_display: int = 20) -> None:
    from matplotlib import transforms

    FS_TICK  = 12
    FS_LABEL = 14
    FS_TITLE = 14

    exp_ord = reindex_explanation(exp, feature_order)
    pretty_names = [LABEL_REPLACEMENTS.get(n, n) for n in exp_ord.feature_names]

    imp = mean_abs_shap_by_feature(exp_ord).reindex(exp_ord.feature_names)
    imp_max = float(imp.max()) if imp.size else 1.0
    imp_map_raw = dict(zip(pretty_names, imp.values))

    sort_flag = False if feature_order else True

    shap.summary_plot(
        exp_ord,
        features=exp_ord.data,
        feature_names=pretty_names,
        plot_type="dot",
        sort=sort_flag,
        show=False,
        plot_size=(fig_size or (8, 10)),
        max_display=max_display,
    )
    ax1 = plt.gca()
    fig = ax1.figure
    fig.canvas.draw()

    yt = [t.get_text() for t in ax1.get_yticklabels()]
    pretty_shown = safe_feature_names(yt if any(yt) else pretty_names)
    ax1.set_yticklabels(pretty_shown, fontsize=FS_TICK)
    ax1.tick_params(axis='x', labelsize=FS_TICK)
    ax1.set_xlabel("SHAP value (impact on model output)", fontsize=FS_LABEL)
    ax1.axvline(0.0, color='0.5', lw=0.8, alpha=0.6)
    for col in ax1.collections:
        col.set_zorder(3)

    ax2 = ax1.twiny()
    shap.summary_plot(
        exp_ord,
        features=exp_ord.data,
        feature_names=pretty_names,
        plot_type="bar",
        sort=sort_flag,
        show=False,
        plot_size=(fig_size or (8, 10)),
        max_display=max_display,
    )
    for b in ax2.patches:
        b.set_alpha(0.22)
        b.set_zorder(0)

    ax2.set_xlim(0, imp_max * 1.05)
    ax2.set_xlabel("Mean |SHAP value| (feature importance)", fontsize=FS_LABEL)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax2.tick_params(axis='x', labelsize=FS_TICK)
    ax2.set_ylim(ax1.get_ylim())
    yhi = ax2.get_ylim()[1]
    ax2.axhline(y=yhi, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)

    cbar_ax = None
    for a in fig.axes:
        if a is ax1 or a is ax2:
            continue
        ylabel = (a.get_ylabel() or "").strip().lower()
        if ylabel.startswith("feature value"):
            cbar_ax = a
            break
    if cbar_ax is not None:
        cbar_ax.tick_params(labelsize=FS_TICK)
        try:
            cbar_ax.yaxis.label.set_fontsize(FS_LABEL)
        except Exception:
            pass

    NUM_COL_FRAC   = 0.06  # 0.08
    GAP_FRAC       = 0.04  # 0.02
    VALUE_FMT      = "{:.2f}"  # "{:.2f}"
    PCT_FMT        = "{:.1f}%"  # "{:.2f}%"
    COLOR_VALUE    = "#2B6CB0"
    COLOR_PCT      = "#C53030"

    ax1_bb = ax1.get_position()
    ax1_x0, ax1_y0, ax1_w, ax1_h = ax1_bb.x0, ax1_bb.y0, ax1_bb.width, ax1_bb.height
    num_w = ax1_w * NUM_COL_FRAC
    gap_w = ax1_w * GAP_FRAC

    if cbar_ax is not None:
        cbar_bb = cbar_ax.get_position()
        gap = cbar_bb.x0 - (ax1_x0 + ax1_w)
        needed = num_w + gap_w
        if gap < needed:
            shrink = needed - gap
            new_w = max(ax1_w - shrink, 0.05)
            ax1.set_position([ax1_x0, ax1_y0, new_w, ax1_h])
            ax2.set_position(ax1.get_position())
            ax1_bb = ax1.get_position()
            ax1_x0, ax1_y0, ax1_w, ax1_h = ax1_bb.x0, ax1_bb.y0, ax1_bb.width, ax1_bb.height
        num_left = cbar_ax.get_position().x0 - (num_w + gap_w)
    else:
        num_left = ax1_x0 + ax1_w + gap_w

    ax_num = fig.add_axes([num_left, ax1_y0, num_w, ax1_h])
    ax_num.patch.set_alpha(0.0)
    ax_num.set_xlim(0, 1)
    ax_num.set_ylim(ax1.get_ylim())
    ax_num.yaxis.set_visible(False)
    ax_num.xaxis.set_visible(False)
    for sp in ax_num.spines.values():
        sp.set_visible(False)

    shown_vals = [float(imp_map_raw.get(n, 0.0)) for n in pretty_shown]
    denom = float(np.sum(shown_vals)) if shown_vals else 1.0

    from matplotlib import transforms
    txt_tr = transforms.blended_transform_factory(ax_num.transAxes, ax1.transData)
    yticks = ax1.get_yticks()
    n_rows = min(len(pretty_shown), len(yticks))
    row_step = abs(yticks[1] - yticks[0]) if n_rows >= 2 else 1.0
    dy = 0.18 * row_step

    pcts = [(100.0 * float(imp_map_raw.get(pretty_shown[i], 0.0)) / denom) if denom > 0 else 0.0
            for i in range(n_rows)]
    if pcts:
        pcts[0] = 100.0 - float(np.sum(pcts[1:]))

    x_left = 0.02
    for i in range(n_rows):
        label = pretty_shown[i]
        y = yticks[i]
        val = float(imp_map_raw.get(label, 0.0))
        ax_num.text(x_left, y + dy, VALUE_FMT.format(val),
                    transform=txt_tr, ha="left", va="center",
                    fontsize=FS_TICK-3, color=COLOR_VALUE)
        ax_num.text(x_left, y - dy, PCT_FMT.format(pcts[i]),
                    transform=txt_tr, ha="left", va="center",
                    fontsize=FS_TICK-3, color=COLOR_PCT)

    plt.suptitle(title, fontsize=FS_TITLE)
    plt.tight_layout()

    if SAVE_PNG:
        plt.savefig(str(out_stem) + ".png", dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        plt.savefig(str(out_stem) + ".svg", format="svg", bbox_inches="tight")
    if SHOW_FIG:
        plt.show()
    plt.close()

# %%
"""
run of plot_combined
"""
print(f"[order] ORDER_MODE={ORDER_MODE}")

# Precompute train order only if needed
train_order = None
if ORDER_MODE == 'train':
    train_pkl = shap_pickle_path("train")
    if not train_pkl.exists():
        raise FileNotFoundError(f"Missing train SHAP at {train_pkl}")
    train_exp = load_explanation(train_pkl)
    train_order = order_from_train(train_exp)
    print(f"[order] Using {len(train_order)} features from train mean(|SHAP|)")

for cohort in COHORTS:  # for cohort in COHORTS, ['train']
    pkl_path = shap_pickle_path(cohort)
    if not pkl_path.exists():
        print(f"[skip] Missing SHAP for {cohort}: {pkl_path}")
        continue

    exp = load_explanation(pkl_path)

    if ORDER_MODE == 'train':
        order_this = ensure_order_subset(train_order, list(exp.feature_names)) or None
    elif ORDER_MODE == 'cohort':
        order_this = list(mean_abs_shap_by_feature(exp).sort_values(ascending=False).index)
    else:
        order_this = None

    if FEATURE_COMB == 'Sleep_Cov':
        pretty_site = SITE_LABEL.get(cohort, cohort)
        concrete_t = concrete_target_for_site(cohort, TARGET)
        title = f"{pretty_site} – {pretty_target_label(concrete_t)}"
        out_stem = OUT_DIR / f"{MODEL}_{TARGET}_{FEATURE_COMB}_SHAP_combined_{cohort}"
        plot_combined(exp, title, out_stem, feature_order=order_this, fig_size=(6.5, 4.4))  # fig_size = (8, 6)
        print(f"[ok] {out_stem}.png")

    if FEATURE_COMB == "Sleep_Cov_Brain":
        MAX_FEATS = 40
        order_40 = (order_this[:MAX_FEATS] if isinstance(order_this, list) else None)
        pretty_site = SITE_LABEL.get(cohort, cohort)
        concrete_t = concrete_target_for_site(cohort, TARGET)
        title = f"{pretty_site} – {pretty_target_label(concrete_t)}"
        out_stem = OUT_DIR / f"{MODEL}_{TARGET}_{FEATURE_COMB}_SHAP_combined_{cohort}"
        plot_combined(exp, title, out_stem, feature_order=order_40, fig_size=(9, 13.5), max_display=MAX_FEATS)  # fig_size = (10, 16)
        print(f"[ok] {out_stem}.png")
