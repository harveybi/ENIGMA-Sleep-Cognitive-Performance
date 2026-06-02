import os
from pathlib import Path
from typing import Optional, Tuple, List
import sys
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib')
import utils  # noqa: F401  (kept for your environment)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# -------------------------------
# Matplotlib / Seaborn style
# -------------------------------
sns.set_context("paper")
mpl.rcParams['font.family']      = 'sans-serif'
mpl.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# %%
# -------------------------------
# Config (edit here interactively)
# -------------------------------
BASE_DIR = "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results"
MODEL = "AutoGluon"
OUT_DIR = "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Results/validation_new_01.06.26"

SAVE_PNG = True  # True or False
SAVE_SVG = True  # True or False
SHOW_FIG = False  # True or False
DPI = 300

# Feature combinations
FEATURES: List[str] = [
    'Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
    'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
    'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle',
    'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle'
]

# Sites → (pretty title, color, folder-name)
SITE_META = {
    'EMC':      ('Rotterdam',  '#F3A3A7', 'EMC'),
    'VETSA':    ('San Diego',  '#E4BDE5', 'VETSA'),
    'Liege':    ('Liège',      '#56BBB8', 'Liege'),
    'KI':       ('Stockholm',  '#7EA8D3', 'KI'),
    'Pitts':    ('Pittsburgh', '#96D3A1', 'Pitts'),
    'Juelich':  ('Jülich',     '#90A2A5', 'Juelich'),
}

# Targets per site
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

# Juelich sessions (each has its own CSV)
JUELICH_SESSIONS = ['sess-1', 'sess-2', 'sess-SD']

# rgo_age rule: allow only for these features
ALLOWED_FOR_RGO_AGE = {'Sleep_Cov', 'Sleep_Cov_Brain'}

# %%
# -------------------------------
# Helpers
# -------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def p_to_superscript(p: float) -> str:
    if p <= 1e-4: return '****'
    if p <= 1e-3: return '***'
    if p <= 1e-2: return '**'
    if p <= 5e-2: return '*'
    return ''

def detect_columns(df: pd.DataFrame, site: Optional[str] = None) -> Tuple[str, str]:
    """
    Resolve (y_true_col, y_pred_col) with preference for '{site}_true'/'{site}_pred'.
    Fallbacks: unique '*_true' & '*_pred', then legacy pairs, then contains('true'/'pred').
    """
    lower2orig = {c.lower(): c for c in df.columns}
    cols_lower = list(lower2orig.keys())

    if site:
        st = f"{site.lower()}_true"
        sp = f"{site.lower()}_pred"
        if st in lower2orig and sp in lower2orig:
            return lower2orig[st], lower2orig[sp]

    trues = [lower2orig[c] for c in cols_lower if c.endswith('_true')]
    preds = [lower2orig[c] for c in cols_lower if c.endswith('_pred')]
    if len(trues) == 1 and len(preds) == 1:
        return trues[0], preds[0]

    for yt, yp in [('y_true','y_pred'),
                   ('true values','predicted values'),
                   ('ytest','ypred'),
                   ('y_test','y_pred')]:
        if yt in lower2orig and yp in lower2orig:
            return lower2orig[yt], lower2orig[yp]

    any_true = [lower2orig[c] for c in cols_lower if 'true' in c]
    any_pred = [lower2orig[c] for c in cols_lower if 'pred' in c]
    if len(any_true) == 1 and len(any_pred) == 1:
        return any_true[0], any_pred[0]

    raise KeyError(f"Cannot detect y_true/y_pred. Columns: {list(df.columns)}")

def read_y_true_pred(csv_path: Path, site: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    y_col, p_col = detect_columns(df, site=site)
    y_true = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
    y_pred = pd.to_numeric(df[p_col], errors='coerce').to_numpy()
    # pairwise finite mask keeps alignment
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]

def is_rgo_age_target(t: str) -> bool:
    return t.endswith('_rgo_age')

def csv_path_for(site_key: str, target: str, feature: str, session: Optional[str]) -> Path:
    base = Path(BASE_DIR) / MODEL
    site_folder = SITE_META[site_key][2]
    if site_key == 'Juelich':
        if session is None:
            raise ValueError("Juelich requires a session.")
        return base / site_folder / session / target / feature / 'results_values.csv'
    return base / site_folder / target / feature / 'results_values.csv'

def output_path_for(site_key: str,
                    target: str,
                    feature: str,
                    pretty_site: str,
                    session: Optional[str]) -> Path:
    """
    Save under OUT_DIR/MODEL/<site_folder>/<target>/<feature>/<PrettySite[_session]>_jointplot.*
    For Juelich, append session to the pretty name.
    """
    site_folder = SITE_META[site_key][2]
    out_dir = Path(OUT_DIR) / MODEL / site_folder / target / feature
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = f"{pretty_site}" if session is None else f"{pretty_site}_{session}"
    return out_dir / f"{stem_name}_jointplot"

def is_stroop_name(site_key: str, t: str) -> bool:
    """Pitts uses 'Executive*'; others use 'Stroop*'."""
    return (t.startswith('Executive') if site_key == 'Pitts' else t.startswith('Stroop'))

def is_memory_name(t: str) -> bool:
    return 'Memory' in t

def classify_target_type(site_key: str, t: str) -> str:
    """
    Returns one of: 'Stroop', 'Memory', 'Stroop_rgo_age', 'Memory_rgo_age'
    Pitts uses 'Executive*' for Stroop variants.
    """
    is_rgo = t.endswith('_rgo_age')
    base = t[:-8] if is_rgo else t  # strip suffix '_rgo_age'
    if site_key == 'Pitts':
        # Stroop-like at Pitts are 'Executive*'
        if base.startswith('Executive'):
            return 'Stroop_rgo_age' if is_rgo else 'Stroop'
    # Everyone else: Stroop*
    if base.startswith('Stroop'):
        return 'Stroop_rgo_age' if is_rgo else 'Stroop'
    # Memory-like (anything containing 'Memory')
    if 'Memory' in base:
        return 'Memory_rgo_age' if is_rgo else 'Memory'
    # Fallback (treat as Memory if unknown but contains Memory later)
    return 'Memory_rgo_age' if is_rgo else 'Memory'

def site_targets_for_type(site_key: str, target_type: str) -> List[str]:
    """
    Return the *concrete* target names at this site that belong to target_type.
    """
    all_ts = TARGETS_BY_SITE.get(site_key, [])
    out: List[str] = []
    for t in all_ts:
        if classify_target_type(site_key, t) == target_type:
            out.append(t)
    return out

def pretty_target_label(t: str) -> str:
    """
    'Memory_Spatial' -> 'Memory Spatial'
    'Stroop_rgo_age' -> 'Stroop (regress out age)'
    'Executive_rgo_age' -> 'Executive (regress out age)'
    """
    if t.endswith('_rgo_age'):
        base = t[:-8]               # strip '_rgo_age'
        base_clean = base.replace('_', ' ')
        return f"{base_clean} (regress out age)"
    return t.replace('_', ' ')

def common_ylim_for_type(target_type: str, feature: str) -> Optional[Tuple[float, float]]:
    """
    Compute a common y-limit for (target_type, feature) across all sites (and Juelich sessions),
    aggregating *predicted* values from the concrete targets that belong to that type.
    Applies the rgo_age feature gating.
    """
    ymin, ymax = np.inf, -np.inf
    found = False
    for site_key in SITE_META.keys():
        concrete_targets = site_targets_for_type(site_key, target_type)
        if not concrete_targets:
            continue

        def ok_feature_for(tname: str) -> bool:
            return (not tname.endswith('_rgo_age')) or (feature in ALLOWED_FOR_RGO_AGE)

        if site_key == 'Juelich':
            for sess in JUELICH_SESSIONS:
                for tname in concrete_targets:
                    if not ok_feature_for(tname):
                        continue
                    csv_p = csv_path_for(site_key, tname, feature, sess)
                    if not csv_p.exists():
                        continue
                    _, yp = read_y_true_pred(csv_p, site=site_key)
                    if yp.size == 0:
                        continue
                    ymin = min(ymin, yp.min()); ymax = max(ymax, yp.max()); found = True
        else:
            for tname in concrete_targets:
                if not ok_feature_for(tname):
                    continue
                csv_p = csv_path_for(site_key, tname, feature, None)
                if not csv_p.exists():
                    continue
                _, yp = read_y_true_pred(csv_p, site=site_key)
                if yp.size == 0:
                    continue
                ymin = min(ymin, yp.min()); ymax = max(ymax, yp.max()); found = True

    if not found:
        return None
    span = ymax - ymin
    pad = 0.05 * span if span > 0 else 1.0
    return (ymin - pad, ymax + pad)

def plot_one_joint(site_key: str,
                   target: str,            # concrete target name
                   feature: str,
                   ylim: Optional[Tuple[float,float]],
                   session: Optional[str] = None):
    pretty_site, scatter_color, _ = SITE_META[site_key]
    csv_p = csv_path_for(site_key, target, feature, session)
    if not csv_p.exists():
        log(f"[skip] Missing CSV: {csv_p}")
        return

    y_true, y_pred = read_y_true_pred(csv_p, site=site_key)
    if y_true.size == 0:
        log(f"[skip] Empty y after filtering: {csv_p}")
        return

    # DataFrame for seaborn
    data = pd.DataFrame({'True Values': y_true, 'Predicted Values': y_pred})

    # Pearson r + p
    # r, p = pearsonr(y_true, y_pred)
    # Spearman r + p
    r, p = spearmanr(y_true, y_pred)
    legend_text = rf"$r = {r:.2f}$" + p_to_superscript(p)  # retain decimal places 2 or 3

    # --- size tuning relative to a 7-inch baseline ---
    BASE_FIG_IN = 7.0
    NEW_FIG_IN = 4.0  # <— set your desired figure width/height here

    scale = NEW_FIG_IN / BASE_FIG_IN

    # scale scatter size by area; linewidths linearly
    s_scaled = max(12, int(50 * (scale ** 2)))  # 50 -> baseline 's' from your code
    edge_lw = max(0.25, 0.5 * scale)  # 0.5 -> baseline edge linewidth
    reg_lw = max(1.5, 4.0 * scale)  # 4.0 -> baseline regression linewidth

    # Jointplot in your style
    # join = sns.jointplot(
    #     x='True Values', y='Predicted Values', data=data, kind='reg',  # height=7,
    #     scatter_kws={'color': scatter_color, 's': 50, 'alpha': 0.8, 'edgecolor': 'black', 'linewidths': 0.5},
    #     line_kws={'color': scatter_color, 'linewidth': 4},
    #     marginal_kws={'color': scatter_color, 'bins': 20, 'fill': True}
    # )
    #
    # # set size 7, 7
    # join.figure.set_size_inches(5, 5)  # width, height 7, 7

    join = sns.jointplot(
        x='True Values', y='Predicted Values', data=data, kind='reg',
        scatter_kws={'color': scatter_color, 's': s_scaled, 'alpha': 0.8,
                     'edgecolor': 'black', 'linewidths': edge_lw},
        line_kws={'color': scatter_color, 'linewidth': reg_lw},
        marginal_kws={'color': scatter_color, 'bins': 20, 'fill': True}
    )

    # set the final figure size in inches (fonts unchanged → appear larger relatively)
    join.figure.set_size_inches(NEW_FIG_IN, NEW_FIG_IN, forward=True)

    # annotate
    join.ax_joint.text(
        0.95, 0.95, legend_text,
        verticalalignment='top', horizontalalignment='right',
        transform=join.ax_joint.transAxes, fontsize=14,
        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3')
    )

    # labels / ticks
    join.set_axis_labels("True Values", "Predicted Values", fontsize=14)
    join.ax_joint.tick_params(axis='both', which='major', labelsize=12)

    # title
    pretty_site, scatter_color, _ = SITE_META[site_key]
    target_label = pretty_target_label(target)
    # fig_title = (f"{pretty_site} – {target_label}" if session is None
    #              else f"{pretty_site} – {target_label} ({session})")
    fig_title = f"{pretty_site}"
    plt.suptitle(fig_title, fontsize=14)

    # enforce common y-limits per (target, feature)
    if ylim is not None:
        join.ax_joint.set_ylim(ylim)

    plt.tight_layout()

    # save to OUT_DIR as configured
    stem = output_path_for(site_key, target, feature, pretty_site, session)
    if SAVE_PNG:
        plt.savefig(str(stem) + ".png", dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        plt.savefig(str(stem) + ".svg", format="svg", bbox_inches="tight")
    if SHOW_FIG:
        plt.show()
    plt.close()

    log(f"[ok] {stem}.png")

def plot_one_lm(site_key: str,
                target: str,            # concrete target name
                feature: str,
                ylim: Optional[Tuple[float, float]],
                session: Optional[str] = None):
    pretty_site, scatter_color, _ = SITE_META[site_key]
    csv_p = csv_path_for(site_key, target, feature, session)
    if not csv_p.exists():
        log(f"[skip] Missing CSV: {csv_p}")
        return

    y_true, y_pred = read_y_true_pred(csv_p, site=site_key)
    if y_true.size == 0:
        log(f"[skip] Empty y after filtering: {csv_p}")
        return

    # DataFrame for seaborn
    data = pd.DataFrame({'True Values': y_true, 'Predicted Values': y_pred})

    # Pearson r + p
    # r, p = r(y_true, y_pred)
    # Spearman r + p
    r, p = spearmanr(y_true, y_pred)
    legend_text = rf"$r = {r:.2f}$" + p_to_superscript(p)

    # --- size tuning relative to a 7-inch baseline ---
    BASE_FIG_IN = 7.0
    NEW_FIG_IN  = 3.0  # set your desired figure width/height here
    scale = NEW_FIG_IN / BASE_FIG_IN

    # scale scatter size by area; linewidths linearly
    s_scaled = max(12, int(50 * (scale ** 2)))   # 50 is your baseline 's'
    edge_lw  = max(0.25, 0.5 * scale)            # 0.5 baseline edge linewidth
    reg_lw   = max(1.5, 4.0 * scale)             # 4.0 baseline regression linewidth

    # lmplot (figure-level), square figure via height + aspect
    g = sns.lmplot(
        data=data, x='True Values', y='Predicted Values', height=NEW_FIG_IN, aspect=1,
        scatter_kws={'color': scatter_color, 's': s_scaled, 'alpha': 0.8,
                     'edgecolor': 'black', 'linewidths': edge_lw},
        line_kws={'color': scatter_color, 'linewidth': reg_lw}
    )

    # be explicit about size as well (helps across seaborn/mpl versions)
    g.figure.set_size_inches(NEW_FIG_IN, NEW_FIG_IN, forward=True)

    # get the single Axes
    ax = getattr(g, "ax", None)
    if ax is None:
        ax = g.axes[0, 0]

    # annotate
    ax.text(
        0.95, 0.95, legend_text,
        va='top', ha='right',
        transform=ax.transAxes, fontsize=14,
        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3')
    )

    # labels / ticks
    g.set_axis_labels("True Values", "Predicted Values", size=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # title
    target_label = pretty_target_label(target)
    # fig_title = (f"{pretty_site} – {target_label}" if session is None
    #              else f"{pretty_site} – {target_label} ({session})")
    fig_title = f"{pretty_site}"
    g.figure.suptitle(fig_title, fontsize=14)

    # enforce common y-limits
    if ylim is not None:
        ax.set_ylim(ylim)

    # tighten layout and leave room for suptitle
    # g.figure.tight_layout(rect=(0, 0, 1, 0.97))
    g.figure.tight_layout()

    # save to OUT_DIR as configured
    stem = output_path_for(site_key, target, feature, pretty_site, session)
    if SAVE_PNG:
        g.figure.savefig(str(stem) + ".png", dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        g.figure.savefig(str(stem) + ".svg", format="svg", bbox_inches="tight")
    if SHOW_FIG:
        plt.show()
    plt.close(g.figure)

    log(f"[ok] {stem}.png")

# %%
TARGET_TYPES = ["Stroop", "Memory", "Stroop_rgo_age", "Memory_rgo_age"]

for target_type in TARGET_TYPES:
    for feature in FEATURES:
        # rgo_age target types require features in ALLOWED_FOR_RGO_AGE
        if target_type.endswith("_rgo_age") and feature not in ALLOWED_FOR_RGO_AGE:
            continue

        # Common y-limits across all sites/sessions for this (target_type, feature)
        ylim = common_ylim_for_type(target_type, feature)

        for site_key in SITE_META.keys():
            concrete_targets = site_targets_for_type(site_key, target_type)
            if not concrete_targets:
                continue

            if site_key == "Juelich":
                for sess in JUELICH_SESSIONS:
                    for tname in concrete_targets:
                        # (defensive) gate rgo_age again at concrete-target level
                        if tname.endswith("_rgo_age") and feature not in ALLOWED_FOR_RGO_AGE:
                            continue
                        # plot_one_joint(site_key, tname, feature, ylim=ylim, session=sess)
                        plot_one_lm(site_key, tname, feature, ylim=ylim, session=sess)
            else:
                for tname in concrete_targets:
                    if tname.endswith("_rgo_age") and feature not in ALLOWED_FOR_RGO_AGE:
                        continue
                    # plot_one_joint(site_key, tname, feature, ylim=ylim)
                    plot_one_lm(site_key, tname, feature, ylim=ylim)
