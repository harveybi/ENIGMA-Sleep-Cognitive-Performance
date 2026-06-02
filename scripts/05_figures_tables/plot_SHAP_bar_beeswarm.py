import os
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
TARGET = ("Stroop")  # 'Stroop' or 'Memory' (set as needed)
FEATURE_COMB = "Sleep_Cov_Brain"  # 'Sleep_Cov' or 'Sleep_Cov_Brain'

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
    # 'SEX': 'SEX',
    # 'BMI': 'BMI',
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
# - 'train': use train mean(|SHAP|) order across all cohorts (current behavior)
# - 'cohort': order by this cohort's mean(|SHAP|) (per-plot ordering)
# - 'none': let shap.summary_plot decide (its own default sort)
ORDER_MODE = 'none'

OUT_DIR = MODEL_DIR / MODEL / TARGET / FEATURE_COMB / "SHAP" / "plots_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Build canonical group sets ---
SLEEP_SET = set(Sleep)
COV_SET = set(Cov)  # | set(APOE4) | set(TIV)
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

# Priority order if a feature occurs in multiple lists (rare but guard it)
BRAIN_GROUPS = {"Thickness", "Area", "Subcortical"}
GROUP_ORDER  = ["Sleep", "Cov", "Thickness", "Area", "Subcortical"]

# Build a deterministic name->group map
NAME_TO_GROUP: dict[str, str] = {}
for g in GROUP_ORDER:
    for name in GROUP_SETS[g]:
        NAME_TO_GROUP.setdefault(name, g)

BAR_COLORS = {
    "Sleep": "#4C78A8",  # blue
    "Cov": "#F58518",  # orange
    "Thickness": "#72B7B2",  # teal
    "Area": "#E45756",  # red
    "Subcortical": "#59A14F",  # green
}
# -------------------------------

# --- Feature count reduction (no L/R collapse) ---
LR_COLLAPSE = False  # keep L/R separate
SELECTION_MODE = 'cohort'   # 'train' | 'cohort'  (basis for selecting Brain features)
CUM_THRESHOLD = 0.85  # keep until this cumulative share (within each Brain group)

# caps only apply to Brain groups; Sleep/Cov are kept in full
MIN_PER_GROUP = {'Thickness':10, 'Area':10, 'Subcortical':10}
MAX_PER_GROUP = {'Thickness':10, 'Area':10, 'Subcortical':10}  # {'Thickness':30, 'Area':30, 'Subcortical':30}

# Plot settings
SAVE_PNG = False
SAVE_SVG = False
SHOW_FIG = True
DPI = 300

# %%
# -------------------------------
# Utils
# -------------------------------
def shap_dir_for(model: str, target: str, feature_comb: str) -> Path:
    return MODEL_DIR / model / target / feature_comb / "SHAP"

def shap_pickle_path(cohort: str) -> Path:
    # train uses the fixed name; test cohorts use suffix
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
    # exp.values: [n_samples, n_features]
    vals = np.asarray(exp.values)
    if vals.ndim != 2:
        # For multioutput, sum abs across outputs
        vals = np.abs(vals).sum(axis=-1)
    mas = np.abs(vals).mean(axis=0)
    return pd.Series(mas, index=list(exp.feature_names))

def order_from_train(train_exp: shap.Explanation) -> List[str]:
    """Descending mean(|SHAP|) order from the train set."""
    s = mean_abs_shap_by_feature(train_exp).sort_values(ascending=False)
    return list(s.index)

def ensure_order_subset(order: List[str], names: List[str]) -> List[str]:
    """Intersect order with available names while preserving order."""
    avail = set(names)
    return [n for n in order if n in avail]

# def reindex_explanation(exp: shap.Explanation, feature_order: List[str]) -> shap.Explanation:
#     """Reorder features of a shap.Explanation to match feature_order."""
#     idx_map = {n: i for i, n in enumerate(exp.feature_names)}
#     cols = [idx_map[n] for n in feature_order]
#     new_vals = np.asarray(exp.values)[:, cols]
#     new_data = None
#     if exp.data is not None and hasattr(exp.data, "shape"):
#         try:
#             new_data = np.asarray(exp.data)[:, cols]
#         except Exception:
#             new_data = exp.data
#     return shap.Explanation(
#         values=new_vals,
#         base_values=exp.base_values,
#         data=new_data,
#         feature_names=feature_order
#     )
def reindex_explanation(exp: shap.Explanation,
                        feature_order: Optional[List[str]]) -> shap.Explanation:
    """Reorder features to match feature_order; if None, return exp unchanged."""
    if not feature_order:
        return exp
    idx_map = {n: i for i, n in enumerate(exp.feature_names)}
    cols = [idx_map[n] for n in feature_order if n in idx_map]
    # if subset is empty or identical, skip
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
    """Readable title: underscores -> spaces; handle rgo suffix if present."""
    if t.endswith('_rgo_age'):
        base = t[:-8].replace('_', ' ')
        return f"{base} (regress out age)"
    return t.replace('_', ' ')

def concrete_target_for_site(site: str, target_type: str) -> str:
    """
    Given a site and a high-level target type ('Stroop'|'Memory'),
    return the concrete target name used at that site for titling.
    Pitts uses 'Executive*' for Stroop. Others use 'Stroop*'.
    For Memory, pick the first non-rgo 'Memory*' available.
    Fallback: the target_type itself.
    """
    if site == 'train':
        return target_type  # training panel uses the generic label

    options = TARGETS_BY_SITE.get(site, [])
    if target_type == 'Stroop':
        prefix = 'Executive' if site == 'Pitts' else 'Stroop'
        choices = [t for t in options if t.startswith(prefix) and not t.endswith('_rgo_age')]
    else:  # 'Memory'
        choices = [t for t in options if ('Memory' in t) and not t.endswith('_rgo_age')]

    return choices[0] if choices else target_type

def feature_group(name: str) -> str:
    """Exact lookup using your lists; fallback = 'Cov' (or change to raise)."""
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
    # Adapted from the PDF. Bins y, offsets points horizontally to avoid overlap. :contentReference[oaicite:2]{index=2}
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
    """
    Returns a long df with columns: name, shap, feature, sample_id, group.
    - shap: signed SHAP value (per sample/feature)
    - feature: raw feature value for color (falls back to NaN if exp.data is None)
    """
    vals = np.asarray(exp.values)   # (n, p)
    n, p = vals.shape
    names = list(exp.feature_names)
    X = np.asarray(exp.data) if (exp.data is not None and hasattr(exp.data, "shape")) else None

    recs = []
    for j, fname in enumerate(names):
        g = feature_group(fname)
        sh_j = vals[:, j]
        feat_j = X[:, j] if X is not None else np.full(n, np.nan)
        # append
        recs.append(pd.DataFrame({
            "name": fname, "group": g,
            "shap": sh_j, "feature": feat_j,
            "sample_id": np.arange(n, dtype=int) + 1
        }))
    df = pd.concat(recs, axis=0, ignore_index=True)
    return df

def select_by_cum_importance(exp: shap.Explanation,
                             cum_threshold: float = CUM_THRESHOLD,
                             max_per_group: dict = MAX_PER_GROUP,
                             min_per_group: dict = MIN_PER_GROUP) -> list[str]:
    """
    Returns selected features:
      - Sleep & Cov: keep ALL
      - Brain groups: keep up to cumulative mean|SHAP| threshold, then ensure a minimum,
        and finally apply per-group caps.
    """
    mas = mean_abs_shap_by_feature(exp).sort_values(ascending=False)

    keep: list[str] = []

    # 1) Keep all Sleep & Cov (preserve MAS order)
    keep.extend([f for f in mas.index if feature_group(f) == "Sleep"])
    keep.extend([f for f in mas.index if feature_group(f) == "Cov"])

    # 2) Brain groups with min / max constraints
    for g in BRAIN_GROUPS:
        idx = [f for f in mas.index if feature_group(f) == g]
        if not idx:
            continue
        s = mas.loc[idx]
        if s.sum() <= 0:
            continue

        # cum-threshold selection
        frac = s / s.sum()
        cum  = frac.cumsum()
        chosen = list(s.index[cum <= cum_threshold])

        # always keep the top one if threshold is tiny and nothing selected
        if not chosen and len(s) > 0:
            chosen = [s.index[0]]

        # enforce minimum (take top importance until min is met)
        min_needed = min_per_group.get(g, 0)
        if len(chosen) < min_needed:
            chosen = list(s.index[:min_needed])

        # apply cap
        cap = max_per_group.get(g)
        if cap is not None:
            chosen = chosen[:cap]

        keep.extend(chosen)

    # Deduplicate while preserving global MAS order
    seen = set()
    ordered_keep = [f for f in mas.index if (f in keep) and (not (f in seen or seen.add(f)))]
    return ordered_keep


# %%
# -------------------------------
# Plot: beeswarm + top-importance bar
# -------------------------------
def plot_combined(exp: shap.Explanation,
                  title: str,
                  out_stem: Path,
                  feature_order: Optional[List[str]]) -> None:
    """
    Draws SHAP beeswarm (dot) and a semi-transparent top bar axis with mean(|SHAP|).
    All derived from the same exp; order fixed by feature_order.
    """
    # Reindex explanation to fixed order
    exp_ord = reindex_explanation(exp, feature_order)

    # pretty names for display only
    pretty_names = [LABEL_REPLACEMENTS.get(n, n) for n in exp_ord.feature_names]

    # Compute per-feature mean(|SHAP|) for the top bar (this cohort)
    imp = mean_abs_shap_by_feature(exp_ord)
    # For nicer axis range (same units as SHAP)
    imp_max = float(imp.max()) if imp.size else 1.0

    # If we gave a fixed order -> sort=False; otherwise let SHAP sort on its own.
    sort_flag = False if feature_order else True

    # 1) Beeswarm
    # shap.summary_plot(
    #     exp_ord,  # Explanation with .data if available
    #     features=exp_ord.data,             # pass data when present for color mapping
    #     # feature_names=exp_ord.feature_names,
    #     feature_names=pretty_names,  # <= use pretty names here
    #     plot_type="dot",
    #     show=False, plot_size=(8, 10), sort=False  # sort=False because we enforce order
    # )
    shap.summary_plot(
        exp_ord,
        features=exp_ord.data,
        feature_names=pretty_names,
        plot_type="dot",
        show=False, plot_size=(8, 10), sort=sort_flag
    )
    ax1 = plt.gca()

    # Replace y tick labels with pretty names
    yt = [t.get_text() for t in ax1.get_yticklabels()]
    pretty = safe_feature_names(yt)
    ax1.set_yticklabels(pretty)
    ax1.set_xlabel("SHAP Value (Impact on Model Output, Beeswarm)")

    # 2) Top axis with bar importance (mean|SHAP|) for the same order
    ax2 = ax1.twiny()
    # Draw invisible bars via shap's bar plot to populate patches on ax2
    # shap.summary_plot(
    #     exp_ord,
    #     features=exp_ord.data,
    #     # feature_names=exp_ord.feature_names,
    #     feature_names=pretty_names,  # <= and here too
    #     plot_type="bar",
    #     show=False, plot_size=(8, 10), sort=False
    # )
    shap.summary_plot(
        exp_ord,
        features=exp_ord.data,
        feature_names=pretty_names,
        plot_type="bar",
        show=False, plot_size=(8, 10), sort=sort_flag
    )
    # Make bars semi-transparent and align axes
    for b in ax2.patches:
        b.set_alpha(0.25)

    ax2.set_xlim(0, imp_max * 1.05)
    ax2.set_xlabel("Mean |SHAP Value| (Feature Importance, Bar)")
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()

    # Cosmetic line near the top labels (optional)
    yhi = ax2.get_ylim()[1]
    ax2.axhline(y=yhi, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)  # y=yhi - 0.5
    # ax2.axhline(
    #     y=1.0, xmin=0, xmax=1, transform=ax2.transAxes,
    #     color='gray', linestyle='-', linewidth=0.8, alpha=0.6
    # )

    # Title
    plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(str(out_stem) + ".png", dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        plt.savefig(str(out_stem) + ".svg", format="svg", bbox_inches="tight")
    if SHOW_FIG:
        plt.show()
    plt.close()

def plot_polar_shap(exp: shap.Explanation,
                    title: str,
                    out_stem: Path,
                    feature_order: Optional[List[str]] = None,
                    figsize=(12, 12)) -> None:
    """
    Polar SHAP: scatter = signed SHAP (radius, shifted), color = feature value,
    inner bars = mean(|SHAP|) per feature (this cohort).
    - feature_order: order around the circle; if None, keep exp.feature_names order.
    """
    # 0) Reindex to desired order (reuse your helper)
    exp_ord = reindex_explanation(exp, feature_order)
    df = long_df_from_explanation(exp_ord)
    names = list(exp_ord.feature_names)
    n_features = len(names)

    # 1) Bar heights = mean(|SHAP|) per feature (this cohort)
    feat_importance = (df.groupby("name")["shap"].apply(lambda s: np.abs(s).mean())
                         .reset_index(name="importance"))
    max_importance = float(feat_importance["importance"].max()) if len(feat_importance) else 1.0

    # 2) Layout params (from PDF, tuned)
    shift = max(1.0, np.percentile(np.abs(df["shap"]), 75))  # ring shift (moves zero outward a bit)
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
    width = angles[1]/2 - angles[1]/10  if n_features > 1 else np.pi/24  # wedge half-width for jitter
    nbins = 60

    cmap = plt.get_cmap("coolwarm")
    feat_min = np.nanmin(df["feature"].values) if np.any(~np.isnan(df["feature"].values)) else 0.0
    feat_max = np.nanmax(df["feature"].values) if np.any(~np.isnan(df["feature"].values)) else 1.0

    # Bars live below the scatter; scale them relative to SHAP spread for separation
    shap_span = (df["shap"].max() - df["shap"].min())
    bar_scale = max(shap_span * 0.6, 1.0)
    bar_bottom = 0.8

    # 3) Start figure
    plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)

    # 4) Draw per-feature scatter and bar (PDF logic) :contentReference[oaicite:3]{index=3}
    for i, name in enumerate(names):
        sub = df[df["name"] == name]

        # angle mid-point between sector i and next sector
        next_idx = (i + 1) % n_features
        angle = (angles[next_idx] - angles[i]) % (2*np.pi)
        base_angle = (angles[i] + angle/2) % (2*np.pi)

        # beeswarm offsets in the small angular window
        x_offsets = simple_beeswarm(sub["shap"].values, nbins=nbins, width=width)
        radius = sub["shap"].values + shift  # shift so 0 is a ring
        # normalize feature values 0..1 for color
        if feat_max > feat_min:
            norm = (sub["feature"].values - feat_min) / (feat_max - feat_min)
        else:
            norm = np.zeros_like(sub["feature"].values)
        colors = cmap(np.clip(norm, 0, 1))

        ax.scatter(base_angle + x_offsets, radius, s=4, alpha=1.0, color=colors, zorder=2)

        # bar (mean|SHAP| for this feature)
        imp = float(feat_importance.loc[feat_importance["name"] == name, "importance"].values[0])
        bar_h = (imp / max_importance) * bar_scale
        color = BAR_COLORS.get(feature_group(name), "#888888")
        ax.bar(base_angle,
               bar_h,
               width=(2*np.pi / n_features) * 0.8,
               bottom=bar_bottom,
               color=color, alpha=1.0, edgecolor="none", zorder=1)

    # 5) Guides / labels as in the PDF (zero ring, outer ring, sector lines, tick rings) :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
    theta = np.linspace(0, 2*np.pi, 360)

    # sector guide + outer labels
    r_max_data = shift + df["shap"].max()
    for i, name in enumerate(names):
        next_idx = (i + 1) % n_features
        angle = (angles[next_idx] - angles[i]) % (2*np.pi)
        label_angle = (angles[i] + angle/2) % (2*np.pi)
        ax.plot([label_angle, label_angle],
                [shift, shift + df["shap"].max()*1.0],
                linestyle='--', lw=0.75, color='#c6c6c6', zorder=1)
        ax.text(label_angle,
                shift + df["shap"].max()*1.15,
                LABEL_REPLACEMENTS.get(name, name),
                ha='center', va='center', fontsize=10, zorder=2)

    # zero ring (SHAP=0)
    ax.plot(theta, np.ones_like(theta)*shift, linestyle='-', lw=1.25, color='#000000', zorder=1)

    # outer border
    r_max = shift + df["shap"].max()*1.0
    ax.plot(theta, np.ones_like(theta)*(r_max + r_max*0.02), linestyle='-', lw=1.25, color='#000000')

    # SHAP radial ticks on left side (π)  :contentReference[oaicite:6]{index=6}
    angle_center = np.deg2rad(180)
    r_min_data, r_max_data = df['shap'].min(), df['shap'].max()
    n_ticks = 6
    neg_ticks = np.linspace(r_min_data, 0, n_ticks//2 + 1, endpoint=False)
    pos_ticks = np.linspace(0, r_max_data, n_ticks//2 + 1)
    r_ticks_data = np.concatenate([neg_ticks, pos_ticks])
    r_ticks = r_ticks_data + shift
    for r in r_ticks:
        ax.plot([angle_center-0.01, angle_center+0.01], [r, r], color='#000000', lw=1, zorder=4)
        ax.text(angle_center, r, f"{r-shift:.1f}",
                ha='center', va='center', fontsize=7, color='#000', fontweight='bold')

    # bar-scale ticks on right side (0°)  :contentReference[oaicite:7]{index=7}
    angle_center = np.deg2rad(0)
    r_min_import, r_max_import = 0.0, max_importance
    tick_vals = np.linspace(r_min_import, r_max_import, 6)
    for rv in tick_vals:
        rr = (rv / max_importance) * bar_scale + bar_bottom
        ax.plot([angle_center-0.01, angle_center+0.01], [rr, rr], color='#000000', lw=1, zorder=4)
        ax.plot(theta, np.ones_like(theta)*rr, linestyle='--', lw=0.75, color='#c6c6c6', zorder=0)
        ax.text(angle_center, rr, f"{rv:.2f}", ha='center', va='center', fontsize=7, fontweight='bold')

    # 6) Legends: feature value colorbar + group legend  :contentReference[oaicite:8]{index=8}
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=feat_min, vmax=feat_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.2)
    cbar.set_label("Feature value", labelpad=-16, y=1.08, rotation=0)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, edgecolor='none', label=g) for g, c in BAR_COLORS.items()]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=9, frameon=False,
              bbox_to_anchor=(1.12, 0.85), title='Group (bars)')

    # 7) Final cosmetics
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(False); ax.spines['polar'].set_visible(False)
    plt.tight_layout()

    # Save/show
    if SAVE_PNG: plt.savefig(str(out_stem) + "_polar.png", dpi=DPI, bbox_inches="tight")
    if SAVE_SVG: plt.savefig(str(out_stem) + "_polar.svg", format="svg", bbox_inches="tight")
    if SHOW_FIG: plt.show()
    plt.close()

# %%
# -------------------------------
# Interactive run (no main)
# -------------------------------
# # 1) Load train explanation for feature ordering
# train_pkl = shap_pickle_path("train")
# if not train_pkl.exists():
#     raise FileNotFoundError(f"Missing train SHAP at {train_pkl}")
# train_exp = load_explanation(train_pkl)
# feature_order = order_from_train(train_exp)
#
# print(f"[order] Using {len(feature_order)} features from train mean(|SHAP|)")
#
# # 2) Per-cohort plots
# for cohort in COHORTS:
#     pkl_path = shap_pickle_path(cohort)
#     if not pkl_path.exists():
#         print(f"[skip] Missing SHAP for {cohort}: {pkl_path}")
#         continue
#
#     exp = load_explanation(pkl_path)
#
#     # use only features present in this cohort’s Explanation (preserving order)
#     order_this = ensure_order_subset(feature_order, list(exp.feature_names))
#     if not order_this:
#         print(f"[skip] No overlapping features with train order for {cohort}")
#         continue
#
#     # pretty_site = SITE_LABEL.get(cohort, cohort)
#     # title = f"{pretty_site} – {TARGET.replace('_', ' ')} ({FEATURE_COMB.replace('_', ', ')})"
#     pretty_site = SITE_LABEL.get(cohort, cohort)
#     concrete_t = concrete_target_for_site(cohort, TARGET)
#     title = f"{pretty_site} – {pretty_target_label(concrete_t)} ({FEATURE_COMB.replace('_', ', ')})"
#
#     out_stem = OUT_DIR / f"{MODEL}_{TARGET}_{FEATURE_COMB}_SHAP_combined_{cohort}"
#
#     plot_combined(exp, title, out_stem, order_this)
#     print(f"[ok] {out_stem}.png")

# %%
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

for cohort in COHORTS:
    pkl_path = shap_pickle_path(cohort)
    if not pkl_path.exists():
        print(f"[skip] Missing SHAP for {cohort}: {pkl_path}")
        continue

    exp = load_explanation(pkl_path)

    if ORDER_MODE == 'train':
        order_this = ensure_order_subset(train_order, list(exp.feature_names))
        # allow empty (falls back to SHAP default sort) by converting to None
        if not order_this:
            order_this = None
    elif ORDER_MODE == 'cohort':
        # order by this cohort's mean(|SHAP|)
        order_this = list(mean_abs_shap_by_feature(exp).sort_values(ascending=False).index)
    else:  # 'none'
        order_this = None

    if FEATURE_COMB == 'Sleep_Cov':
        pretty_site = SITE_LABEL.get(cohort, cohort)
        concrete_t = concrete_target_for_site(cohort, TARGET)
        title = f"{pretty_site} – {pretty_target_label(concrete_t)}"  # "{pretty_site} – {pretty_target_label(concrete_t)} ({FEATURE_COMB.replace('_', ', ')})"
        out_stem = OUT_DIR / f"{MODEL}_{TARGET}_{FEATURE_COMB}_SHAP_combined_{cohort}"

        plot_combined(exp, title, out_stem, order_this)
        print(f"[ok] {out_stem}.png")

    if FEATURE_COMB == "Sleep_Cov_Brain":
        if SELECTION_MODE == 'train':
            # build selection on train Explanation (loaded once)
            if 'train_exp_for_sel' not in globals():
                tp = shap_pickle_path("train")
                if not tp.exists():
                    raise FileNotFoundError(f"Missing train SHAP at {tp}")
                train_exp_for_sel = load_explanation(tp)
            sel_basis_exp = train_exp_for_sel
        else:
            sel_basis_exp = exp

        selected = select_by_cum_importance(sel_basis_exp)
        # enforce selection while preserving current ordering policy
        base_order = order_this or list(exp.feature_names)  # ORDER_MODE respected upstream
        order_this = [f for f in base_order if f in selected] or selected or None

        # (optional) diagnostics
        def _count_by_group(names):
            from collections import Counter
            return Counter(feature_group(n) for n in names)

        print("[select] kept =", len(order_this) if order_this else 0,
              "by group:", _count_by_group(order_this or []))

        pretty_site = SITE_LABEL.get(cohort, cohort)
        concrete_t = concrete_target_for_site(cohort, TARGET)
        title = f"{pretty_site} – {pretty_target_label(concrete_t)} (Sleep, Demo, Brain)"
        out_stem = OUT_DIR / f"{MODEL}_{TARGET}_{FEATURE_COMB}_SHAP_{cohort}"
        plot_polar_shap(exp, title, out_stem, feature_order=order_this)

# %%
# feature combination is Sleep_Cov_Brain


