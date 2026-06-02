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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter

from pyvirtualdisplay import Display

# --- NEW: surfplot/brainspace stack ---
from neuromaps.datasets import fetch_fslr
from brainspace.datasets import load_conte69, load_parcellation
from surfplot import Plot
from brainspace.utils.parcellation import map_to_labels
from brainspace.mesh.mesh_io import read_surface as _bs_read_surface
from brainspace.plotting.surface_plotting import plot_surf as _bs_plot_surf
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_surf_fsaverage
import nibabel as nib

# -------------------------------
# Headless display helper (safe no-op if DISPLAY is present)
# -------------------------------
@contextlib.contextmanager
def _virtual_display(size=(1600, 1200)):
    disp = None
    need_vdisplay = (os.environ.get("DISPLAY") in (None, "", ":0") or os.environ.get("CI") == "true")
    if need_vdisplay:
        disp = Display(visible=0, size=size)  # requires Xvfb on the server
        disp.start()
    try:
        yield
    finally:
        if disp is not None:
            disp.stop()

# -------------------------------
# Style
# -------------------------------
sns.set_context("paper")
mpl.rcParams['font.family']      = 'sans-serif'
mpl.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %%
# -------------------------------
# Config
# -------------------------------
feature_lists_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/feature_lists.pkl'
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
MODEL  = "AutoGluon"      # 'AutoGluon' or 'XGBoost'
TARGET = ("Stroop")         # 'Stroop' or 'Memory'
FEATURE_COMB = "Sleep_Cov_Brain"  # 'Sleep_Cov' or 'Sleep_Cov_Brain'

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

BAR_COLORS = {
    "Sleep": "#4C78A8",
    "Cov": "#F58518",
    "Thickness": "#72B7B2",
    "Area": "#E45756",
    "Subcortical": "#59A14F",
}

# --- Feature count reduction (no L/R collapse) ---
LR_COLLAPSE = False
SELECTION_MODE = 'cohort'
CUM_THRESHOLD = 0.85
MIN_PER_GROUP = {'Thickness':10, 'Area':10, 'Subcortical':10}
MAX_PER_GROUP = {'Thickness':10, 'Area':10, 'Subcortical':10}

# Plot settings
SAVE_PNG = True
SAVE_SVG = True
SHOW_FIG = False
DPI = 300

# ---- Canonical DK order (34 per hemisphere) ----
_DK34 = [
    "bankssts","caudalanteriorcingulate","caudalmiddlefrontal","cuneus","entorhinal",
    "fusiform","inferiorparietal","inferiortemporal","isthmuscingulate","lateraloccipital",
    "lateralorbitofrontal","lingual","medialorbitofrontal","middletemporal","parahippocampal",
    "paracentral","parsopercularis","parsorbitalis","parstriangularis","pericalcarine",
    "postcentral","posteriorcingulate","precentral","precuneus","rostralanteriorcingulate",
    "rostralmiddlefrontal","superiorfrontal","superiorparietal","superiortemporal",
    "supramarginal","frontalpole","temporalpole","transversetemporal","insula"
]

_DK_SYNONYMS = {
    "bankssts":"bankssts", "banks_of_superior_temporal_sulcus":"bankssts",
    "caudalanteriorcingulate":"caudalanteriorcingulate", "caudal_anterior_cingulate":"caudalanteriorcingulate",
    "caudalmiddlefrontal":"caudalmiddlefrontal", "caudal_middle_frontal":"caudalmiddlefrontal",
    "isthmuscingulate":"isthmuscingulate", "isthmus_cingulate":"isthmuscingulate",
    "lateralorbitofrontal":"lateralorbitofrontal", "lateral_orbitofrontal":"lateralorbitofrontal",
    "medialorbitofrontal":"medialorbitofrontal", "medial_orbitofrontal":"medialorbitofrontal",
    "middletemporal":"middletemporal", "middle_temporal":"middletemporal",
    "parstriangularis":"parstriangularis", "pars_triangularis":"parstriangularis",
    "parsopercularis":"parsopercularis", "pars_opercularis":"parsopercularis",
    "parahippocampal":"parahippocampal", "paracentral":"paracentral",
    "posteriorcingulate":"posteriorcingulate", "posterior_cingulate":"posteriorcingulate",
    "rostralanteriorcingulate":"rostralanteriorcingulate", "rostral_anterior_cingulate":"rostralanteriorcingulate",
    "rostralmiddlefrontal":"rostralmiddlefrontal", "rostral_middle_frontal":"rostralmiddlefrontal",
    "superiorfrontal":"superiorfrontal", "superior_frontal":"superiorfrontal",
    "superiorparietal":"superiorparietal", "superior_parietal":"superiorparietal",
    "superiortemporal":"superiortemporal", "superior_temporal":"superiortemporal",
    "lateraloccipital":"lateraloccipital", "lateral_occipital":"lateraloccipital",
    "transversetemporal":"transversetemporal", "transverse_temporal":"transversetemporal",
}

# DK_CSV_DIR = Path("~").expanduser() / "Projects/ENIGMA/ENIGMA/enigmatoolbox/datasets/parcellations"
# DK_CSV_PATH = DK_CSV_DIR / "aparc_conte69.csv"
DK_CSV_PATH = Path("/home/h.bi/Projects/ENIGMA/ENIGMA/enigmatoolbox/datasets/parcellations/aparc_conte69.csv")

# ---- where the ENIGMA subcortex meshes live (you already set these) ----
SCTX_DIR = Path('/home/h.bi/Projects/ENIGMA/ENIGMA/enigmatoolbox/datasets/surfaces')
SCTX_LH = SCTX_DIR / "sctx_lh.gii"
SCTX_RH = SCTX_DIR / "sctx_rh.gii"

# ---- canonical order expected by ENIGMA plots (16 entries) ----
SUBCORT16_ORDER = [
    "L_accumbens","L_amygdala","L_caudate","L_hippocampus",
    "L_pallidum","L_putamen","L_thalamus","L_ventricles",
    "R_accumbens","R_amygdala","R_caudate","R_hippocampus",
    "R_pallidum","R_putamen","R_thalamus","R_ventricles",
]

# ---- robust aliases from your Aseg names to these 16 buckets ----
_ASEG_ALIASES = {
    # Left
    "left-accumbens-area": "L_accumbens",
    "left-accumbens":      "L_accumbens",
    "left-amygdala":       "L_amygdala",
    "left-caudate":        "L_caudate",
    "left-hippocampus":    "L_hippocampus",
    "left-pallidum":       "L_pallidum",
    "left-pallidun":       "L_pallidum",
    "left-putamen":        "L_putamen",
    "left-thalamus":       "L_thalamus",
    "left-lateral-ventricle":   "L_ventricles",
    "left-inf-lat-vent":        "L_ventricles",

    # Right
    "right-accumbens-area":"R_accumbens",
    "right-accumbens":     "R_accumbens",
    "right-amygdala":      "R_amygdala",
    "right-caudate":       "R_caudate",
    "right-hippocampus":   "R_hippocampus",
    "right-pallidum":      "R_pallidum",
    "right-pallidun":      "R_pallidum",
    "right-putamen":       "R_putamen",
    "right-thalamus":      "R_thalamus",
    "right-lateral-ventricle":  "R_ventricles",
    "right-inf-lat-vent":       "R_ventricles",

    # If you want midline ventricles to contribute to both sides, uncomment:
    # "3rd-ventricle":  ("L_ventricles","R_ventricles"),
    # "4th-ventricle":  ("L_ventricles","R_ventricles"),
    # "5th-ventricle":  ("L_ventricles","R_ventricles"),
}

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

    NUM_COL_FRAC   = 0.09
    GAP_FRAC       = 0.03
    VALUE_FMT      = "{:,.3f}"
    PCT_FMT        = "{:.3f}%"
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
                    fontsize=FS_TICK-2, color=COLOR_VALUE)
        ax_num.text(x_left, y - dy, PCT_FMT.format(pcts[i]),
                    transform=txt_tr, ha="left", va="center",
                    fontsize=FS_TICK-2, color=COLOR_PCT)

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
# -------------------------------
# Atlas-name parsing & vectors
# -------------------------------
def _canon_feat_base(name: str) -> str:
    # drop trailing metric suffixes
    return re.sub(r'_(thickness|area|meancurv|gauscurv|volume)$', '', name, flags=re.I)

def _extract_hemi_and_token(name: str) -> tuple[str|None, str|None]:
    """
    From 'LH_Cont_Cing_1_thickness' -> ('lh', 'cont_cing_1')
    Robust to extra separators/case. Returns (hemi, token) or (None, None)
    """
    base = _canon_feat_base(name)
    m = re.search(r'\b(lh|rh)[_\-\.](.+)$', base, flags=re.I)
    if not m:
        return None, None
    hemi = m.group(1).lower()
    tail = m.group(2)
    m2 = re.search(r'([A-Za-z]+(?:_[A-Za-z]+)*)_(\d{1,3})$', tail)
    if not m2:
        return None, None
    token = (m2.group(1) + '_' + m2.group(2)).lower()
    return hemi, token

def build_schaefer_name_lut_from_nilearn(scale: int = 400, networks: int = 7) -> dict[str, dict[str, int]]:
    """
    Returns:
      {'lh': {'cont_cing_1': 123, ...}, 'rh': {...}}
    where values are 1..200 indices within each hemisphere in the SAME order
    used by nilearn's Schaefer 2018 atlas.
    """
    atlas = fetch_atlas_schaefer_2018(n_rois=scale, yeo_networks=networks)
    labels = list(atlas['labels'])
    lut = {'lh': {}, 'rh': {}}
    idx_lh = 0
    idx_rh = 0
    for lab in labels:
        if lab.lower().startswith('background'):
            continue
        # '7Networks_LH_Cont_Cing_1'
        parts = lab.split('_')
        if 'LH' in parts:
            hemi = 'lh'
            i = parts.index('LH')
        elif 'RH' in parts:
            hemi = 'rh'
            i = parts.index('RH')
        else:
            continue
        token = '_'.join(parts[i+1:]).lower()  # e.g., 'cont_cing_1'
        if hemi == 'lh':
            idx_lh += 1
            lut['lh'][token] = idx_lh
        else:
            idx_rh += 1
            lut['rh'][token] = idx_rh
    # quick sanity check
    if len(lut['lh']) != scale//2 or len(lut['rh']) != scale//2:
        print(f"[warn] LUT sizes unexpected: LH={len(lut['lh'])}, RH={len(lut['rh'])}")
    return lut

# Build once at import/runtime
_SCHAEFER_NAME_LUT = build_schaefer_name_lut_from_nilearn(scale=400, networks=7)

def _canonize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _dk_token_from_name(name: str) -> str | None:
    t = _canonize_token(name)
    t = re.sub(r"(thickness|area|thk|surf|surface|meancurv|gauscurv|volume)$", "", t)
    t = re.sub(r"^(lh|rh|left|right|l|r)[\._-]*", "", t)
    return _DK_SYNONYMS.get(t, t)

def _extract_hemi(name: str) -> str | None:
    m = re.search(r"\b(lh|rh|left|right|l|r)\b", name.lower())
    if not m:
        m = re.match(r"^(lh|rh)", name.lower())
    if not m:
        return None
    h = m.group(1)
    return "lh" if h in ("lh","left","l") else "rh"

def _assemble_dk_vector(series: pd.Series, expect_metric: str) -> np.ndarray:
    """DK68 (LH 34 + RH 34). Kept here for completeness but we won't render DK via surfplot unless you want it."""
    vals = np.zeros(68, dtype=float)
    seen = set()
    unmatched = []
    for feat, v in series.items():
        hemi = _extract_hemi(feat) or ("lh" if "left" in feat.lower() else "rh" if "right" in feat.lower() else None)
        if hemi is None:
            unmatched.append(feat); continue
        token = _dk_token_from_name(feat)
        if token not in _DK34:
            unmatched.append(feat); continue
        idx_in_hemi = _DK34.index(token)
        idx = idx_in_hemi if hemi == "lh" else 34 + idx_in_hemi
        vals[idx] = float(v)
        seen.add((hemi, token))
    if unmatched:
        print(f"[warn] {len(unmatched)} DK features not matched to parcels (showing up to 8):")
        for u in unmatched[:8]:
            print("   -", u)
    if len(seen) < 68:
        print(f"[info] DK coverage: {len(seen)}/68 parcels mapped; missing parcels filled with 0.")
    return vals

def _schaefer_series_from_exp(exp: shap.Explanation, fam: str) -> pd.Series:
    """
    Return mean|SHAP| for Schaefer features in the requested family:
      fam ∈ {'Thickness', 'Area'}
    """
    mas = mean_abs_shap_by_feature(exp)
    cand = [n for n in mas.index if feature_group(n) == fam]
    def _is_schaefer_label(name: str) -> bool:
        t = name.lower()
        if "schaefer" in t or "sch400" in t or "400parc" in t:
            return True
        if re.search(r"\b(7|17)\s*net", t):
            return True
        # common 7-network tokens seen in names like LH_Cont_Cing_1_thickness
        for tok in ("vis","sommot","dorsattn","salventattn","limbic","cont","default"):
            if f"_{tok}_" in t:
                return True
        return False
    keep = [n for n in cand if _is_schaefer_label(n)]
    s = mas.loc[keep].sort_index()
    if s.empty:
        print(f"[warn] No Schaefer features found for family '{fam}'.")
    return s

_SCH_NUM_RE = re.compile(r"(?:^|[_\-\.])(lh|rh)[^0-9]*?(\d{1,3})(?:$|[^0-9])", re.IGNORECASE)

def _assemble_schaefer_vector(series: pd.Series, n_per_hemi: int = 200, agg: str = "max") -> np.ndarray:
    """
    Build length-400 vector: LH[1..200], RH[1..200] using a robust name→index LUT.
    If a feature cannot be found in the LUT, we fall back to extracting a bare number,
    which can cause collisions (reported).
    agg: 'max' | 'mean' | 'sum' to combine duplicate features hitting the same parcel.
    """
    vals = np.zeros(2 * n_per_hemi, dtype=float)
    counts: dict[tuple[str,int], list[float]] = {}
    unmatched, via_lut, via_numeric = [], 0, 0

    for feat, v in series.items():
        hemi, token = _extract_hemi_and_token(feat)
        used = False
        if hemi and token and token in _SCHAEFER_NAME_LUT[hemi]:
            idx = _SCHAEFER_NAME_LUT[hemi][token]
            counts.setdefault((hemi, idx), []).append(float(v))
            via_lut += 1
            used = True
        if not used:
            # fallback: bare number (often WRONG with network-scoped names)
            m = _SCH_NUM_RE.search(feat)
            if m:
                hemi2 = "lh" if m.group(1).lower() == "lh" else "rh"
                idx2  = int(m.group(2))
                if 1 <= idx2 <= n_per_hemi:
                    counts.setdefault((hemi2, idx2), []).append(float(v))
                    via_numeric += 1
                    used = True
        if not used:
            unmatched.append(feat)

    # aggregate duplicates per (hemi, idx)
    dups = []
    seen_L, seen_R = set(), set()
    for (hemi, idx), vals_at_idx in counts.items():
        if len(vals_at_idx) > 1:
            dups.append((hemi, idx, len(vals_at_idx)))
        if agg == "sum":
            val = float(np.sum(vals_at_idx))
        elif agg == "mean":
            val = float(np.mean(vals_at_idx))
        else:
            val = float(np.max(vals_at_idx))
        ofs = 0 if hemi == "lh" else n_per_hemi
        vals[ofs + (idx - 1)] = val
        (seen_L if hemi == "lh" else seen_R).add(idx)

    uniqL, uniqR = len(seen_L), len(seen_R)
    total_unique = uniqL + uniqR
    print(f"\n[info] Schaefer unique coverage: LH {uniqL}/200 + RH {uniqR}/200 = {total_unique}/{2*n_per_hemi} "
          f"indices filled (agg='{agg}'; via_lut={via_lut}, via_numeric={via_numeric}, unmatched={len(unmatched)})")
    if dups:
        ex = ", ".join([f"{h}{i}×{n}" for (h,i,n) in dups[:10]])
        more = f" (+{len(dups)-10} more)" if len(dups) > 10 else ""
        print(f"[warn] {len(dups)} parcel-index collisions after aggregation. Examples: {ex}{more}")
    if unmatched:
        print(f"[warn] {len(unmatched)} features could not be matched at all (first 10):")
        for u in unmatched[:10]:
            print("   -", u)
    return vals

def _vector_to_vertex_data_fslr(vec_400: np.ndarray):
    """
    Map a Schaefer-400 vector (LH 1..200, RH 1..200) to fsLR (Conte69) vertex data
    and return ((lh_surf, rh_surf), {'left': lh_data, 'right': rh_data}).
    Also prints diagnostics about NaNs and non-zero coverage.
    """
    fslr = fetch_fslr()
    lh_surf, rh_surf = fslr["inflated"]   # 'midthickness' also ok

    labL, labR = load_parcellation('schaefer', scale=400, join=False)
    labL = np.asarray(labL, dtype=int)
    labR = np.asarray(labR, dtype=int)

    n_per_hemi = 200
    vec_400 = np.asarray(vec_400, dtype=float).ravel()
    assert vec_400.size == 2 * n_per_hemi, f"Expected 400 values, got {vec_400.size}"

    valsL = vec_400[:n_per_hemi]
    valsR = vec_400[n_per_hemi:]

    lutL = np.zeros(int(labL.max()) + 1, dtype=float); lutL[1:1 + n_per_hemi] = valsL
    lutR = np.zeros(int(labR.max()) + 1, dtype=float); lutR[1:1 + n_per_hemi] = valsR

    lh_data = map_to_labels(lutL, labL, mask=labL > 0, fill=np.nan).astype(float)
    rh_data = map_to_labels(lutR, labR, mask=labR > 0, fill=np.nan).astype(float)

    # ---- diagnostics ----
    for hemi, arr, lab in (("LH", lh_data, labL), ("RH", rh_data, labR)):
        n = arr.size
        n_nan = int(np.isnan(arr).sum())
        n_fin = n - n_nan
        n_zero = int(np.isfinite(arr).sum() - np.count_nonzero(np.nan_to_num(arr)))
        n_pos  = int(np.count_nonzero(np.nan_to_num(arr)))
        lab_covered = int((lab > 0).sum())
        print(f"[diag] {hemi}: vertices={n}, labeled={lab_covered}, finite={n_fin}, NaN={n_nan}, "
              f"zeros={n_zero}, >0={n_pos}, min={np.nanmin(arr):.4g}, median={np.nanmedian(arr):.4g}, "
              f"max={np.nanmax(arr):.4g}")

    return (lh_surf, rh_surf), {'left': lh_data, 'right': rh_data}

_SCH7_TOKENS = ("vis","sommot","dorsattn","salventattn","limbic","cont","default")
def _is_schaefer_label(name: str) -> bool:
    t = name.lower()
    if "schaefer" in t or "sch400" in t or "sch400x7" in t or "400parc" in t:
        return True
    if any(f"_{tok}_" in t or t.startswith(f"lh_{tok}_") or t.startswith(f"rh_{tok}_") for tok in _SCH7_TOKENS):
        return True
    if re.search(r"\b(7|17)\s*net", t):
        return True
    return False

def _mas_series_for(exp: shap.Explanation, which: str) -> pd.Series:
    mas = mean_abs_shap_by_feature(exp)
    w = which.lower()
    fam = "Thickness" if "thickness" in w else "Area" if "area" in w else None
    if fam is None:
        raise ValueError(f"Unknown 'which': {which}")
    cand = [n for n in mas.index if feature_group(n) == fam]
    if "dk" in w:
        keep = [n for n in cand if not _is_schaefer_label(n)]
    else:
        keep = [n for n in cand if _is_schaefer_label(n)]
    s = mas.loc[keep].sort_index()
    if s.empty:
        print(f"[warn] No features matched for {which}.")
    return s

# %%
# -------------------------------
# Surfplot mapping & rendering (Schaefer-400 on Conte69)
# -------------------------------
def _vector_to_vertex_data_schaefer_conte69(vec_2hemi: np.ndarray, scale: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Map a 2*N Schaefer parcel vector (LH 1..N, RH 1..N) onto Conte69 vertices.
    Returns (lh_vertex_data, rh_vertex_data).
    """
    if vec_2hemi.ndim != 1:
        vec_2hemi = np.asarray(vec_2hemi).ravel()

    # Get Schaefer labels for Conte69 (per hemisphere)
    labL, labR = load_parcellation('schaefer', scale=scale, join=False)  # <-- no 'networks' kwarg

    n_per_hemi = scale // 2
    assert vec_2hemi.size == 2 * n_per_hemi, f"Expected vector of length {2*n_per_hemi}, got {vec_2hemi.size}"

    vecL = vec_2hemi[:n_per_hemi]
    vecR = vec_2hemi[n_per_hemi:]

    # Build dense label->value tables (index 0 = background)
    srcL = np.zeros(int(labL.max()) + 1, dtype=float)
    srcR = np.zeros(int(labR.max()) + 1, dtype=float)
    # Most Schaefer labelings use label IDs 1..N per hemi; fill by position.
    # If some label IDs are missing, those stay at 0.
    fillL = min(n_per_hemi, srcL.size - 1)
    fillR = min(n_per_hemi, srcR.size - 1)
    srcL[1:fillL + 1] = vecL[:fillL]
    srcR[1:fillR + 1] = vecR[:fillR]

    # Map to vertices (NaN for medial wall / background)
    lh_data = map_to_labels(srcL, labL, mask=labL > 0, fill=np.nan)
    rh_data = map_to_labels(srcR, labR, mask=labR > 0, fill=np.nan)
    return lh_data, rh_data

def _assemble_schaefer_vector_from_names(series: pd.Series, n_per_hemi: int = 200) -> np.ndarray:
    """
    Best-effort builder for a 2*N vector from feature names. If your feature names
    contain only within-network indices (e.g., LH_Cont_Cing_1), we cannot recover
    the global 1..N parcel ID deterministically without a LUT; we then fall back to
    extracting any trailing number and place it as that parcel index. This yields
    partial coverage. For perfect coverage, provide numeric IDs in names (e.g., ..._L_123).
    """
    vals = np.zeros(2 * n_per_hemi, dtype=float)
    unmatched = []

    # numeric form: ... (lh|rh)[^0-9]*?(\d{1,3}) ...
    num_re = re.compile(r"(?:^|[_\-\.])(lh|rh)[^0-9]*?(\d{1,3})(?:$|[^0-9])", re.IGNORECASE)

    for feat, v in series.items():
        m = num_re.search(feat)
        if not m:
            unmatched.append(feat)
            continue
        hemi = "lh" if m.group(1).lower() == "lh" else "rh"
        idx = int(m.group(2))  # expects global 1..200 within each hemi
        if 1 <= idx <= n_per_hemi:
            ofs = 0 if hemi == "lh" else n_per_hemi
            vals[ofs + (idx - 1)] = float(v)
        else:
            unmatched.append(feat)

    nz = int(np.count_nonzero(vals))
    if unmatched:
        print(f"[warn] {len(unmatched)} Schaefer features could not be matched to global parcel IDs (showing up to 8):")
        for u in unmatched[:8]:
            print("   -", u)
    print(f"[info] Schaefer coverage: {nz}/{2*n_per_hemi} parcels mapped; missing parcels filled with 0.")
    return vals

def _mas_series_for_schaefer(exp: shap.Explanation, which: str) -> pd.Series:
    """
    Extract mean|SHAP| series for 'Thickness_Schaefer' or 'Area_Schaefer'.
    """
    mas = mean_abs_shap_by_feature(exp)
    which_low = which.lower()
    fam = "Thickness" if "thickness" in which_low else "Area"
    # Keep only Schaefer features (heuristic by name)
    def _is_schaefer_label(name: str) -> bool:
        t = name.lower()
        return ("schaefer" in t) or bool(re.search(r"(?:^|[_\-\.])(lh|rh).*?(?:\d{1,3})(?:$|[^0-9])", t))
    keep = [n for n in mas.index if feature_group(n) == fam and _is_schaefer_label(n)]
    s = mas.loc[keep].sort_index()
    if s.empty:
        print(f"[warn] No Schaefer features found for '{which}'.")
    return s

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def _subcort_series_from_exp(exp: shap.Explanation) -> pd.Series:
    mas = mean_abs_shap_by_feature(exp)
    keep = [n for n in mas.index if feature_group(n) == "Subcortical"]
    s = mas.loc[keep].sort_index()
    if s.empty:
        print("[warn] No subcortical features found.")
    return s

def _assemble_subcortical_vector16(series: pd.Series,
                                   agg: str = "mean",
                                   share_midline: bool = True) -> np.ndarray:
    """
    Map Aseg mean|SHAP| into the 16-element ENIGMA ordering.
    - agg: 'mean' | 'max' when multiple features feed the same bucket (e.g., ventricles).
    - share_midline: if midline aliases are present, contribute to L/R ventricles.
    """
    buckets: dict[str, list[float]] = {k: [] for k in SUBCORT16_ORDER}
    unmatched = []

    for raw_name, v in series.items():
        key = _norm(raw_name)
        target = _ASEG_ALIASES.get(key)
        if target is None:
            unmatched.append(raw_name)
            continue
        if isinstance(target, tuple):
            if share_midline:
                for t in target:
                    buckets[t].append(float(v))
        else:
            buckets[target].append(float(v))

    vec = []
    filled = 0
    for k in SUBCORT16_ORDER:
        vals = buckets[k]
        if not vals:
            vec.append(0.0)
        else:
            filled += 1
            vec.append(float(np.mean(vals) if agg == "mean" else np.max(vals)))

    print(f"[subctx] mapped {filled}/16 targets (agg='{agg}'), unmatched={len(unmatched)}")
    if unmatched:
        print("         unmatched examples:", ", ".join(unmatched[:6]), ("..." if len(unmatched) > 6 else ""))
    return np.asarray(vec, float)

def plot_surfplot_cortical_from_shap(exp: shap.Explanation,
                                     which: str,
                                     out_png: Path,
                                     cmap: str = "Blues_r",
                                     color_range: Tuple[float, float] | None = None,
                                     size: Tuple[int, int] = (1600, 800)) -> None:
    """
    Render cortical surface with surfplot for:
      which ∈ {'Thickness_Schaefer','Area_Schaefer'}
    Saves PNG to out_png. DK variants are skipped (no robust fsLR DK mapping here).
    """
    w = which.lower()
    if "schaefer" not in w:
        print(f"[skip] {which}: only Schaefer surfaces are supported in surfplot version; skipping.")
        return

    fam = "Thickness" if "thickness" in w else "Area" if "area" in w else None
    if fam is None:
        print(f"[skip] {which}: unknown family; skipping.")
        return

    s = _schaefer_series_from_exp(exp, fam=fam)
    if s.empty:
        print(f"[skip] {which}: no features after Schaefer filtering; skipping.")
        return

    vec400 = _assemble_schaefer_vector(s, n_per_hemi=200, agg="max")
    if not np.any(vec400):
        print(f"[skip] {which}: vector is all zeros after mapping; skipping.")
        return

    (lh_surf, rh_surf), data_lr = _vector_to_vertex_data_fslr(vec400)

    with _virtual_display(size=(max(size[0], 1200), max(size[1], 600))):
        p = Plot(lh_surf, rh_surf, size=size, zoom=1.2, layout='row')
        # pass dict {'left': ..., 'right': ...} to avoid "Data type invalid"
        p.add_layer(data=data_lr, cmap=cmap, cbar=False, color_range=color_range)  # cbar=True
        fig = p.build()
        fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[ok] saved cortical surface (surfplot fsLR) -> {out_png}")

# --- DK (aparc) on fsLR/Conte69 via ENIGMA-Toolbox ---
def _canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def _gifti_labels_and_names(gii_path: Path) -> tuple[np.ndarray, dict[int, str]]:
    g = nib.load(str(gii_path))
    lab = np.asarray(g.agg_data()).astype(int)
    name_by_id = {}
    lt = getattr(g, 'labeltable', None)
    if lt is not None and getattr(lt, 'labels', None):
        for L in lt.labels:
            name_by_id[int(L.key)] = _canon(L.label)
    return lab, name_by_id

def _find_enigma_dk_fslr_label_files() -> tuple[Path, Path]:
    """
    Try to locate ENIGMA-Toolbox DK (aparc) label.gii files on fsLR/Conte69.
    If not found, raise with a clear instruction.
    """
    try:
        import enigmatoolbox as _enigma
    except Exception:
        raise RuntimeError(
            "ENIGMA-Toolbox not found. Install it (e.g., `pip install ENIGMA-Toolbox`) "
            "so we can load DK (aparc) labels on fsLR/Conte69."
        )
    base = Path(_enigma.__file__).parent / "datasets" / "parcellations"
    # Look for label.gii under any conte/fslr folder containing 'aparc' or 'desikan'
    left_cands  = list(base.rglob("*conte*/*aparc*/*lh*.label.gii")) + list(base.rglob("*fslr*/*aparc*/*lh*.label.gii")) + \
                  list(base.rglob("*conte*/*desikan*/*lh*.label.gii")) + list(base.rglob("*fslr*/*desikan*/*lh*.label.gii"))
    right_cands = list(base.rglob("*conte*/*aparc*/*rh*.label.gii")) + list(base.rglob("*fslr*/*aparc*/*rh*.label.gii")) + \
                  list(base.rglob("*conte*/*desikan*/*rh*.label.gii")) + list(base.rglob("*fslr*/*desikan*/*rh*.label.gii"))
    if not left_cands or not right_cands:
        raise RuntimeError(
            f"Could not find DK fsLR label.gii files under {base}. "
            f"If your ENIGMA-Toolbox layout differs, set env var DK_FSLR_DIR to a folder that contains "
            f"`lh*.label.gii` and `rh*.label.gii` for DK on fsLR."
        )
    # pick the first matching pair (you can refine later if you have multiple versions)
    return Path(left_cands[0]), Path(right_cands[0])

def _load_dk_labels_from_enigma_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    if not csv_path.exists():
        raise FileNotFoundError(f"DK CSV not found: {csv_path}")

    # fsLR vert counts to split single-column files safely
    fslr = fetch_fslr()
    lh_surf_path, rh_surf_path = fslr["inflated"]
    nL = nib.load(lh_surf_path).darrays[0].data.shape[0]
    nR = nib.load(rh_surf_path).darrays[0].data.shape[0]

    df_raw = pd.read_csv(csv_path, header=None)
    df_num = df_raw.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna(axis=0, how="any")

    if df_num.shape[1] >= 2:
        labL = df_num.iloc[:, 0].astype(int).to_numpy()
        labR = df_num.iloc[:, 1].astype(int).to_numpy()
    elif df_num.shape[1] == 1:
        vec = df_num.iloc[:, 0].astype(int).to_numpy().ravel()
        if vec.size == (nL + nR):
            labL, labR = vec[:nL], vec[nL:]
        elif vec.size % 2 == 0:
            mid = vec.size // 2
            print(f"[dk-csv] WARN: length {vec.size} != nL+nR ({nL+nR}). Using even split {mid}+{mid}.")
            labL, labR = vec[:mid], vec[mid:]
        else:
            raise ValueError(f"Single-col DK CSV length {vec.size} doesn’t match fsLR nL+nR ({nL+nR}).")
    else:
        raise ValueError(f"Expected 1 or 2 numeric columns in {csv_path}, got {df_num.shape[1]}")

    uL = np.unique(labL[labL > 0]); uR = np.unique(labR[labR > 0])
    print(f"\n[dk-csv] {csv_path.name}: verts L={labL.size}, R={labR.size} | "
          f"unique-ID count L={uL.size}, R={uR.size}")

    return labL.astype(int), labR.astype(int)

def _pick_34_ids_per_hemi(lab: np.ndarray, hemi_tag: str) -> list[int]:
    """Choose the 34 cortical IDs for one hemisphere from a label map."""
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    # Heuristic: if IDs look offset (all > 34), normalize by subtracting min-1
    norm_ids = ids.copy()
    offset = 0
    if norm_ids.min() > 34:
        offset = int(norm_ids.min() - 1)
        norm_ids = norm_ids - offset

    # Build table for decision/debug
    table = sorted([(int(i), int(c), int(i - offset)) for i, c in zip(ids, counts)], key=lambda t: t[0])

    # If we have more than 34 IDs, drop the one with the fewest vertices (usually an extra code)
    choose = list(ids)
    if len(choose) > 34:
        drop_id, drop_cnt = min(zip(ids, counts), key=lambda t: t[1])
        choose.remove(drop_id)
        print(f"[dk-map:{hemi_tag}] {len(ids)} IDs -> dropping extra id={drop_id} (count={drop_cnt})")

    # If still not 34, fail loudly
    if len(choose) != 34:
        raise RuntimeError(f"[dk-map:{hemi_tag}] Need 34 cortical IDs, got {len(choose)}. "
                           f"IDs={ids.tolist()} (offset={offset}).")

    # Return IDs sorted by their *normalized* value so 1..34 ordering is preserved
    choose_sorted = [i for (_, i) in sorted([(int(i - offset), int(i)) for i in choose], key=lambda t: t[0])]
    print(f"[dk-map:{hemi_tag}] offset={offset} | first 5 chosen IDs: {choose_sorted[:5]}")
    return choose_sorted

def _vector_to_vertex_data_fslr_dk(vec_68: np.ndarray, dk_csv_path: Path):
    """
    Map DK 68-vector to fsLR vertices using ENIGMA aparc_conte69.csv.
    We DO NOT assume IDs are 1..34; we discover the 34 cortical IDs per hemi and
    assign vec positions (1..34) to those IDs in normalized order.
    """
    fslr = fetch_fslr()
    lh_surf, rh_surf = fslr["inflated"]

    labL, labR = _load_dk_labels_from_enigma_csv(dk_csv_path)

    # pick the 34 cortical IDs per hemisphere & build LUTs
    idsL = _pick_34_ids_per_hemi(labL, "LH")
    idsR = _pick_34_ids_per_hemi(labR, "RH")

    # vec_68 = [LH 34 | RH 34] in your canonical _DK34 order
    vec_68 = np.asarray(vec_68, float).ravel()
    assert vec_68.size == 68, f"Expected 68 values, got {vec_68.size}"
    valsL = vec_68[:34]
    valsR = vec_68[34:]

    max_id = int(max(labL.max(), labR.max()))
    lutL = np.zeros(max_id + 1, float)
    lutR = np.zeros(max_id + 1, float)

    # Assign by discovered ID order (normalized 1..34)
    for pos, lab_id in enumerate(idsL, start=1):
        if pos <= 34: lutL[lab_id] = valsL[pos-1]
    for pos, lab_id in enumerate(idsR, start=1):
        if pos <= 34: lutR[lab_id] = valsR[pos-1]

    lh_data = lutL[labL]
    rh_data = lutR[labR]

    # Diags
    nposL = int(np.count_nonzero(lh_data)); nposR = int(np.count_nonzero(rh_data))
    print(f"[diag-DK-fsLR] LH>0={nposL}, RH>0={nposR}, "
          f"minL={np.nanmin(lh_data):.5g}, maxL={np.nanmax(lh_data):.5g}, "
          f"minR={np.nanmin(rh_data):.5g}, maxR={np.nanmax(rh_data):.5g}")

    return (lh_surf, rh_surf), {'left': lh_data, 'right': rh_data}

def plot_surfplot_cortical_from_shap_dk(exp: shap.Explanation,
                                        which: str,
                                        out_png: Path,
                                        cmap: str = "Blues_r",
                                        color_range: Tuple[float, float] | None = None,
                                        size: Tuple[int, int] = (1600, 800)) -> None:
    w = which.lower()
    if "dk" not in w:
        print(f"[skip] {which}: this DK renderer expects '*_DK'."); return
    fam = "Thickness" if "thickness" in w else "Area" if "area" in w else None
    if fam is None:
        print(f"[skip] {which}: unknown family; skipping."); return

    s = _mas_series_for(exp, which=which)
    if s.empty:
        print(f"[skip] {which}: no DK features after filtering; skipping."); return

    vec68 = _assemble_dk_vector(s, expect_metric=fam)
    if not np.any(vec68):
        print(f"[skip] {which}: DK vector is all zeros; skipping."); return

    (lh_surf, rh_surf), data_lr = _vector_to_vertex_data_fslr_dk(vec68, DK_CSV_PATH)

    with _virtual_display(size=(max(size[0], 1200), max(size[1], 600))):
        p = Plot(lh_surf, rh_surf, size=size, zoom=1.2, layout='row')
        p.add_layer(data=data_lr, cmap=cmap, cbar=False, color_range=color_range)  # cbar=True
        fig = p.build()
        fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[ok] saved cortical surface (surfplot fsLR, DK) -> {out_png}")

# -------- ENIGMA subcortex helpers (adapted; no external dependency) --------
def subcorticalvertices(subcortical_values=None):
    """
    Expand 16 region values into per-vertex data for the ENIGMA subcortex mesh.
    Order must be:
    L_accumbens, L_amygdala, L_caudate, L_hippocampus, L_pallidum, L_putamen, L_thalamus, L_ventricles,
    R_accumbens, R_amygdala, R_caudate, R_hippocampus, R_pallidum, R_putamen, R_thalamus, R_ventricles
    """
    numvertices = [867,1419,3012,3784,1446,4003,3726,7653,  838,1457,3208,3742,1373,3871,3699,7180]
    data = []
    if isinstance(subcortical_values, np.ndarray):
        for ii in range(16):
            data.append(np.tile(subcortical_values[ii], (numvertices[ii], 1)))
        data = np.vstack(data).flatten()
    return data

def plot_subcortical(array_name=None,
                     ventricles=True,
                     color_bar=False,
                     color_range=None,
                     cmap="Reds",
                     nan_color=(1,1,1,0),
                     zoom=1.3,
                     background=(1,1,1),
                     size=(1200,300),
                     interactive=False,
                     embed_nb=True,
                     screenshot=True,
                     filename=None,
                     transparent_bg=True,
                     scale=(1,1),
                     **kwargs):

    if not (SCTX_LH.exists() and SCTX_RH.exists()):
        raise FileNotFoundError(f"sctx meshes not found at {SCTX_LH} / {SCTX_RH}")

    surf_lh = _bs_read_surface(str(SCTX_LH))
    surf_rh = _bs_read_surface(str(SCTX_RH))
    n_pts_lh, n_pts_rh = surf_lh.n_points, surf_rh.n_points
    n_pts_tot = n_pts_lh + n_pts_rh

    layout4 = ["lh","lh","rh","rh"]
    view4   = ["lateral","medial","lateral","medial"]
    surfs   = {"lh": surf_lh, "rh": surf_rh}

    def _ensure_vertex_vec(vec_or_regionvals):
        """Accept a 16/14-len region vector OR an already-vertex vector; return vertex vec len==n_pts_tot."""
        v = np.asarray(vec_or_regionvals)
        if v.ndim != 1:
            raise ValueError("Only 1D vectors are accepted here.")
        if v.size == 16 and ventricles:
            return np.asarray(subcorticalvertices(v))
        if v.size == 14 and not ventricles:
            tmp = np.empty(16); tmp[:] = np.nan
            tmp[:7]  = v[:7]
            tmp[8:15] = v[7:]
            return np.asarray(subcorticalvertices(tmp))
        if v.size == n_pts_tot:
            return v
        raise ValueError(f"Unexpected vector length {v.size}; "
                         f"expected 16 (ventricles=True), 14 (ventricles=False), or {n_pts_tot} vertices.")

    def _append_vec_to_meshes(vec):
        """Append per-vertex data to LH/RH and return the array name (string)."""
        name = surf_lh.append_array(vec[:n_pts_lh], at="p")
        surf_rh.append_array(vec[n_pts_lh:], name=name, at="p")
        return name

    # --- normalize array_name into “names on the meshes” ---
    layout = [layout4]           # default 1 row
    name_arg = None              # string or (rows×1) array of strings

    if isinstance(array_name, pd.Series):
        array_name = array_name.to_numpy()

    if isinstance(array_name, np.ndarray) and array_name.ndim == 1:
        # single map
        vec = _ensure_vertex_vec(array_name)
        name_arg = _append_vec_to_meshes(vec)  # <- pass a string to plot_surf

    elif isinstance(array_name, np.ndarray) and array_name.ndim == 2:
        # multiple rows
        names = []
        for row in array_name:
            vec = _ensure_vertex_vec(row)
            names.append(_append_vec_to_meshes(vec))
        layout = [layout4] * len(names)
        name_arg = np.asarray(names)[:, None]  # rows×1 → broadcast across 4 columns

    elif isinstance(array_name, list):
        names = []
        for item in array_name:
            vec = _ensure_vertex_vec(item)
            names.append(_append_vec_to_meshes(vec))
        layout = [layout4] * len(names)
        name_arg = np.asarray(names)[:, None]

    else:
        raise ValueError("array_name must be a 1D vector, a 2D array (rows), or a list of vectors.")

    # render
    return _bs_plot_surf(
        surfs,
        layout,
        array_name=name_arg,
        color_bar=('right' if color_bar else False),
        color_range=color_range,
        cmap=cmap,
        nan_color=nan_color,
        view=view4,
        share="r",
        zoom=zoom,
        background=background,
        size=size,
        interactive=interactive,
        embed_nb=embed_nb,
        screenshot=screenshot,
        filename=filename,
        transparent_bg=transparent_bg,
        scale=scale,
        **kwargs,
    )

def plot_subcortical_from_shap(exp: shap.Explanation,
                               out_png: Path,
                               cmap: str = "Reds",
                               color_range: tuple[float,float] | None = None,
                               agg: str = "mean",
                               size: tuple[int,int] = (1200, 300)) -> None:
    """
    Build 16-vector from SHAP and render to PNG via ENIGMA subcortex meshes.
    """
    s = _subcort_series_from_exp(exp)
    if s.empty:
        print("[skip] subcortical: no features after filtering; skip.")
        return

    v16 = _assemble_subcortical_vector16(s, agg=agg, share_midline=True)

    with _virtual_display(size=(max(size[0], 800), max(size[1], 300))):
        plot_subcortical(
            array_name=v16,
            ventricles=True,
            size=size,
            zoom=1.3,
            color_range=color_range,
            cmap=cmap,
            embed_nb=True,
            screenshot=True,
            filename=str(out_png),
            transparent_bg=True,
        )
    print(f"[ok] saved subcortical surface -> {out_png}")

def compute_uniform_brain_colorrange(exp: shap.Explanation,
                                     schaefer_agg: str = "max",
                                     subcort_agg: str = "mean") -> tuple[float, float]:
    """
    Compute a single color range (0..vmax) for all brain plots from this SHAP Explanation.
    - Uses the same vector builders you plot with, so the range reflects your actual maps:
      * Schaefer (thickness/area): agg='max' by default (matches your plotting)
      * DK (thickness/area): direct mapping
      * Subcortical (Aseg): agg='mean' by default (matches your plotting)
    """
    vmax_candidates: list[float] = []

    # Schaefer THK
    s = _schaefer_series_from_exp(exp, fam="Thickness")
    if not s.empty:
        v = _assemble_schaefer_vector(s, n_per_hemi=200, agg=schaefer_agg)
        vmax_candidates.append(np.nanmax(v))

    # Schaefer AREA
    s = _schaefer_series_from_exp(exp, fam="Area")
    if not s.empty:
        v = _assemble_schaefer_vector(s, n_per_hemi=200, agg=schaefer_agg)
        vmax_candidates.append(np.nanmax(v))

    # DK THK
    s = _mas_series_for(exp, which="Thickness_DK")
    if not s.empty:
        v = _assemble_dk_vector(s, expect_metric="Thickness")
        vmax_candidates.append(np.nanmax(v))

    # DK AREA
    s = _mas_series_for(exp, which="Area_DK")
    if not s.empty:
        v = _assemble_dk_vector(s, expect_metric="Area")
        vmax_candidates.append(np.nanmax(v))

    # Subcortical (Aseg)
    s = _subcort_series_from_exp(exp)
    if not s.empty:
        v16 = _assemble_subcortical_vector16(s, agg=subcort_agg, share_midline=True)
        vmax_candidates.append(np.nanmax(v16))

    # Finalize
    vmax = float(np.nanmax(vmax_candidates)) if len(vmax_candidates) else 0.0
    # avoid degenerate range (0,0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-12
    print(f"[cm-range] uniform color range = (0, {vmax:.6g})")
    return (0.0, vmax)

def compute_uniform_brain_colorrange_simple(
    exp: shap.Explanation,
    groups: list[list[str]] = None,
) -> tuple[float, float]:
    """
    Return a single color range (0..vmax) where vmax is the maximum
    mean|SHAP| among all *brain* features.
    """
    if groups is None:
        groups = [Thickness_DK, Thickness_Schaefer, Area_DK, Area_Schaefer, Subcortical]

    # union of all brain features
    brain_feats = set().union(*groups)

    # mean|SHAP| per feature
    mas = mean_abs_shap_by_feature(exp)

    # keep only brain features that are present in exp
    keep = [f for f in mas.index if f in brain_feats]
    if not keep:
        print("[cm-range] No brain features found in this Explanation; using tiny range.")
        return (0.0, 1e-12)

    vmax = float(mas.loc[keep].max())
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-12

    print(f"[cm-range] uniform color range = (0, {vmax:.6g}) from {len(keep)} brain features")
    return (0.0, vmax)

def truncate_cmap(cmap='Reds', start=0.30, end=1.0, n=256):
    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    new = LinearSegmentedColormap.from_list(
        f'{getattr(base, "name", "cmap")}_trunc_{start:.2f}_{end:.2f}',
        base(np.linspace(start, end, n))
    )
    return new
CMAP_BRAIN = truncate_cmap('Reds', start=0.35, end=1.0)


def make_vertical_colorbar(
    cmap,
    color_range: tuple[float, float],
    label: str = "Mean |SHAP|",
    height_px: int = 800,
    width_px: int = 140,
    n_ticks: int = 6,
    tick_fmt: str = "{x:.3g}",
    outfile: Optional[str] = None,
    dpi: int = 300,
    labelpad: int = 24,   # <-- add
    tickpad: int = 6,     # <-- add
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    vmin, vmax = color_range
    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    ax = fig.add_axes([0.35, 0.05, 0.30, 0.90])  # you can widen to 0.32–0.35 if needed
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapped_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mapped_cmap)
    sm.set_array([])

    cb = plt.colorbar(sm, cax=ax, orientation='vertical')
    cb.set_label(label, rotation=90, va='center', labelpad=labelpad)   # <-- padding
    cb.ax.tick_params(pad=tickpad)                                     # <-- gap ticks↔bar

    if n_ticks is not None:
        cb.locator = MaxNLocator(n_ticks)
        cb.update_ticks()
    if tick_fmt is not None:
        cb.formatter = FuncFormatter(lambda x, pos: tick_fmt.format(x=x))
        cb.update_ticks()

    if outfile:
        fig.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# %%
"""
run of plot_combined
"""
print(f"\n[order] ORDER_MODE={ORDER_MODE}")

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
        print(f("[skip] Missing SHAP for {cohort}: {pkl_path}"))
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
        plot_combined(exp, title, out_stem, order_this)
        print(f"[ok] {out_stem}.png")

    if FEATURE_COMB == "Sleep_Cov_Brain":
        MAX_FEATS = 40
        order_40 = (order_this[:MAX_FEATS] if isinstance(order_this, list) else None)
        pretty_site = SITE_LABEL.get(cohort, cohort)
        concrete_t = concrete_target_for_site(cohort, TARGET)
        title = f"{pretty_site} – {pretty_target_label(concrete_t)} (Sleep, Cov, Brain)"
        out_stem = OUT_DIR / f"{MODEL}_{TARGET}_{FEATURE_COMB}_SHAP_combined_{cohort}"
        plot_combined(exp, title, out_stem, feature_order=order_40, fig_size=(10, 16), max_display=MAX_FEATS)
        print(f"[ok] {out_stem}.png")

        uniform_cr = compute_uniform_brain_colorrange(exp)
        # uniform_cr = compute_uniform_brain_colorrange_simple(exp)

        cbar_dir = OUT_DIR / "surfaces_surfplot"
        cbar_dir.mkdir(parents=True, exist_ok=True)

        make_vertical_colorbar(
            cmap=CMAP_BRAIN,  # or 'Reds' to match your current calls
            color_range=uniform_cr,
            label="Mean |SHAP|",
            height_px=800,
            width_px=140,
            n_ticks=6,
            tick_fmt="{x:.3g}",
            outfile=str(cbar_dir / f"{MODEL}_{TARGET}_{cohort}_colorbar.png"),
        )

        # ---- Surfplot cortical renders (Schaefer only, as requested) ----
        surf_dir = OUT_DIR / "surfaces_surfplot"
        surf_dir.mkdir(parents=True, exist_ok=True)

        plot_surfplot_cortical_from_shap(
            exp, which="Thickness_Schaefer",
            out_png=surf_dir / f"{MODEL}_{TARGET}_{cohort}_Thickness_Schaefer.png",
            cmap="Reds", color_range=uniform_cr, size=(1600, 800)  # cmap="Reds"
        )
        plot_surfplot_cortical_from_shap(
            exp, which="Area_Schaefer",
            out_png=surf_dir / f"{MODEL}_{TARGET}_{cohort}_Area_Schaefer.png",
            cmap="Reds", color_range=uniform_cr, size=(1600, 800)  # cmap="Reds"
        )

        # NEW: DK (Desikan–Killiany)
        plot_surfplot_cortical_from_shap_dk(
            exp, which="Thickness_DK",
            out_png=surf_dir / f"{MODEL}_{TARGET}_{cohort}_Thickness_DK.png",
            cmap="Reds", color_range=uniform_cr, size=(1600, 800)  # cmap="Reds"
        )
        plot_surfplot_cortical_from_shap_dk(
            exp, which="Area_DK",
            out_png=surf_dir / f"{MODEL}_{TARGET}_{cohort}_Area_DK.png",
            cmap="Reds", color_range=uniform_cr, size=(1600, 800)  # cmap="Reds"
        )

        plot_subcortical_from_shap(
            exp,
            out_png=surf_dir / f"{MODEL}_{TARGET}_{cohort}_Subcortex_Aseg.png",
            cmap="Reds",  # cmap="Reds"
            color_range=uniform_cr,  # or (vmin, vmax) to lock across cohorts
            agg="mean",
            size=(1200, 300),
        )

# %%
# =============================== SUBCORTICAL BLOCK (self-contained) ===============================
# from pathlib import Path
# import re
# import numpy as np
# import pandas as pd
#
# # brainspace I/O + plotting
# from brainspace.mesh.mesh_io import read_surface as _bs_read_surface
# from brainspace.plotting.surface_plotting import plot_surf as _bs_plot_surf
#
# # ---- where the ENIGMA subcortex meshes live (you already set these) ----
# SCTX_DIR = Path('/home/h.bi/Projects/ENIGMA/ENIGMA/enigmatoolbox/datasets/surfaces')
# SCTX_LH = SCTX_DIR / "sctx_lh.gii"
# SCTX_RH = SCTX_DIR / "sctx_rh.gii"
#
# # ---- canonical order expected by ENIGMA plots (16 entries) ----
# SUBCORT16_ORDER = [
#     "L_accumbens","L_amygdala","L_caudate","L_hippocampus",
#     "L_pallidum","L_putamen","L_thalamus","L_ventricles",
#     "R_accumbens","R_amygdala","R_caudate","R_hippocampus",
#     "R_pallidum","R_putamen","R_thalamus","R_ventricles",
# ]
#
# # ---- robust aliases from your Aseg names to these 16 buckets ----
# _ASEG_ALIASES = {
#     # Left
#     "left-accumbens-area": "L_accumbens",
#     "left-accumbens":      "L_accumbens",
#     "left-amygdala":       "L_amygdala",
#     "left-caudate":        "L_caudate",
#     "left-hippocampus":    "L_hippocampus",
#     "left-pallidum":       "L_pallidum",
#     "left-pallidun":       "L_pallidum",
#     "left-putamen":        "L_putamen",
#     "left-thalamus":       "L_thalamus",
#     "left-lateral-ventricle":   "L_ventricles",
#     "left-inf-lat-vent":        "L_ventricles",
#
#     # Right
#     "right-accumbens-area":"R_accumbens",
#     "right-accumbens":     "R_accumbens",
#     "right-amygdala":      "R_amygdala",
#     "right-caudate":       "R_caudate",
#     "right-hippocampus":   "R_hippocampus",
#     "right-pallidum":      "R_pallidum",
#     "right-pallidun":      "R_pallidum",
#     "right-putamen":       "R_putamen",
#     "right-thalamus":      "R_thalamus",
#     "right-lateral-ventricle":  "R_ventricles",
#     "right-inf-lat-vent":       "R_ventricles",
#
#     # If you want midline ventricles to contribute to both sides, uncomment:
#     # "3rd-ventricle":  ("L_ventricles","R_ventricles"),
#     # "4th-ventricle":  ("L_ventricles","R_ventricles"),
#     # "5th-ventricle":  ("L_ventricles","R_ventricles"),
# }
#
# def _norm(s: str) -> str:
#     return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
#
# def _subcort_series_from_exp(exp: shap.Explanation) -> pd.Series:
#     mas = mean_abs_shap_by_feature(exp)
#     keep = [n for n in mas.index if feature_group(n) == "Subcortical"]
#     s = mas.loc[keep].sort_index()
#     if s.empty:
#         print("[warn] No subcortical features found.")
#     return s
#
# def _assemble_subcortical_vector16(series: pd.Series,
#                                    agg: str = "mean",
#                                    share_midline: bool = True) -> np.ndarray:
#     """
#     Map Aseg mean|SHAP| into the 16-element ENIGMA ordering.
#     - agg: 'mean' | 'max' when multiple features feed the same bucket (e.g., ventricles).
#     - share_midline: if midline aliases are present, contribute to L/R ventricles.
#     """
#     buckets: dict[str, list[float]] = {k: [] for k in SUBCORT16_ORDER}
#     unmatched = []
#
#     for raw_name, v in series.items():
#         key = _norm(raw_name)
#         target = _ASEG_ALIASES.get(key)
#         if target is None:
#             unmatched.append(raw_name)
#             continue
#         if isinstance(target, tuple):
#             if share_midline:
#                 for t in target:
#                     buckets[t].append(float(v))
#         else:
#             buckets[target].append(float(v))
#
#     vec = []
#     filled = 0
#     for k in SUBCORT16_ORDER:
#         vals = buckets[k]
#         if not vals:
#             vec.append(0.0)
#         else:
#             filled += 1
#             vec.append(float(np.mean(vals) if agg == "mean" else np.max(vals)))
#
#     print(f"[subctx] mapped {filled}/16 targets (agg='{agg}'), unmatched={len(unmatched)}")
#     if unmatched:
#         print("         unmatched examples:", ", ".join(unmatched[:6]), ("..." if len(unmatched) > 6 else ""))
#     return np.asarray(vec, float)
#
# # -------- ENIGMA subcortex helpers (adapted; no external dependency) --------
# def subcorticalvertices(subcortical_values=None):
#     """
#     Expand 16 region values into per-vertex data for the ENIGMA subcortex mesh.
#     Order must be:
#     L_accumbens, L_amygdala, L_caudate, L_hippocampus, L_pallidum, L_putamen, L_thalamus, L_ventricles,
#     R_accumbens, R_amygdala, R_caudate, R_hippocampus, R_pallidum, R_putamen, R_thalamus, R_ventricles
#     """
#     numvertices = [867,1419,3012,3784,1446,4003,3726,7653,  838,1457,3208,3742,1373,3871,3699,7180]
#     data = []
#     if isinstance(subcortical_values, np.ndarray):
#         for ii in range(16):
#             data.append(np.tile(subcortical_values[ii], (numvertices[ii], 1)))
#         data = np.vstack(data).flatten()
#     return data
#
# def plot_subcortical(array_name=None,
#                      ventricles=True,
#                      color_bar=False,
#                      color_range=None,
#                      cmap="Reds",
#                      nan_color=(1,1,1,0),
#                      zoom=1.3,
#                      background=(1,1,1),
#                      size=(1200,300),
#                      interactive=False,
#                      embed_nb=True,
#                      screenshot=True,
#                      filename=None,
#                      transparent_bg=True,
#                      scale=(1,1),
#                      **kwargs):
#
#     if not (SCTX_LH.exists() and SCTX_RH.exists()):
#         raise FileNotFoundError(f"sctx meshes not found at {SCTX_LH} / {SCTX_RH}")
#
#     surf_lh = _bs_read_surface(str(SCTX_LH))
#     surf_rh = _bs_read_surface(str(SCTX_RH))
#     n_pts_lh, n_pts_rh = surf_lh.n_points, surf_rh.n_points
#     n_pts_tot = n_pts_lh + n_pts_rh
#
#     layout4 = ["lh","lh","rh","rh"]
#     view4   = ["lateral","medial","lateral","medial"]
#     surfs   = {"lh": surf_lh, "rh": surf_rh}
#
#     def _ensure_vertex_vec(vec_or_regionvals):
#         """Accept a 16/14-len region vector OR an already-vertex vector; return vertex vec len==n_pts_tot."""
#         v = np.asarray(vec_or_regionvals)
#         if v.ndim != 1:
#             raise ValueError("Only 1D vectors are accepted here.")
#         if v.size == 16 and ventricles:
#             return np.asarray(subcorticalvertices(v))
#         if v.size == 14 and not ventricles:
#             tmp = np.empty(16); tmp[:] = np.nan
#             tmp[:7]  = v[:7]
#             tmp[8:15] = v[7:]
#             return np.asarray(subcorticalvertices(tmp))
#         if v.size == n_pts_tot:
#             return v
#         raise ValueError(f"Unexpected vector length {v.size}; "
#                          f"expected 16 (ventricles=True), 14 (ventricles=False), or {n_pts_tot} vertices.")
#
#     def _append_vec_to_meshes(vec):
#         """Append per-vertex data to LH/RH and return the array name (string)."""
#         name = surf_lh.append_array(vec[:n_pts_lh], at="p")
#         surf_rh.append_array(vec[n_pts_lh:], name=name, at="p")
#         return name
#
#     # --- normalize array_name into “names on the meshes” ---
#     layout = [layout4]           # default 1 row
#     name_arg = None              # string or (rows×1) array of strings
#
#     if isinstance(array_name, pd.Series):
#         array_name = array_name.to_numpy()
#
#     if isinstance(array_name, np.ndarray) and array_name.ndim == 1:
#         # single map
#         vec = _ensure_vertex_vec(array_name)
#         name_arg = _append_vec_to_meshes(vec)  # <- pass a string to plot_surf
#
#     elif isinstance(array_name, np.ndarray) and array_name.ndim == 2:
#         # multiple rows
#         names = []
#         for row in array_name:
#             vec = _ensure_vertex_vec(row)
#             names.append(_append_vec_to_meshes(vec))
#         layout = [layout4] * len(names)
#         name_arg = np.asarray(names)[:, None]  # rows×1 → broadcast across 4 columns
#
#     elif isinstance(array_name, list):
#         names = []
#         for item in array_name:
#             vec = _ensure_vertex_vec(item)
#             names.append(_append_vec_to_meshes(vec))
#         layout = [layout4] * len(names)
#         name_arg = np.asarray(names)[:, None]
#
#     else:
#         raise ValueError("array_name must be a 1D vector, a 2D array (rows), or a list of vectors.")
#
#     # render
#     return _bs_plot_surf(
#         surfs,
#         layout,
#         array_name=name_arg,
#         color_bar=('right' if color_bar else False),
#         color_range=color_range,
#         cmap=cmap,
#         nan_color=nan_color,
#         view=view4,
#         share="r",
#         zoom=zoom,
#         background=background,
#         size=size,
#         interactive=interactive,
#         embed_nb=embed_nb,
#         screenshot=screenshot,
#         filename=filename,
#         transparent_bg=transparent_bg,
#         scale=scale,
#         **kwargs,
#     )
#
# def plot_subcortical_from_shap(exp: shap.Explanation,
#                                out_png: Path,
#                                cmap: str = "Reds",
#                                color_range: tuple[float,float] | None = None,
#                                agg: str = "mean",
#                                size: tuple[int,int] = (1200, 300)) -> None:
#     """
#     Build 16-vector from SHAP and render to PNG via ENIGMA subcortex meshes.
#     """
#     s = _subcort_series_from_exp(exp)
#     if s.empty:
#         print("[skip] subcortical: no features after filtering; skip.")
#         return
#
#     v16 = _assemble_subcortical_vector16(s, agg=agg, share_midline=True)
#
#     with _virtual_display(size=(max(size[0], 800), max(size[1], 300))):
#         plot_subcortical(
#             array_name=v16,
#             ventricles=True,
#             size=size,
#             zoom=1.3,
#             color_range=color_range,
#             cmap=cmap,
#             embed_nb=True,
#             screenshot=True,
#             filename=str(out_png),
#             transparent_bg=True,
#         )
#     print(f"[ok] saved subcortical surface -> {out_png}")
# # =========================== END SUBCORTICAL BLOCK ===========================
#
# for cohort in COHORTS:
#     pkl = shap_pickle_path(cohort)
#     if not pkl.exists():
#         print(f"[skip] Missing SHAP for {cohort}: {pkl}")
#         continue
#
#     exp = load_explanation(pkl)
#
#     surf_dir = OUT_DIR / "surfaces_surfplot"
#     surf_dir.mkdir(parents=True, exist_ok=True)
#
#     plot_subcortical_from_shap(
#         exp,
#         out_png=surf_dir / f"{MODEL}_{TARGET}_{cohort}_Subcortex_Aseg.png",
#         cmap="Reds",
#         color_range=None,  # or (vmin, vmax) to lock across cohorts
#         agg="mean",
#         size=(1200, 300),
#     )

# %%
# 2D subcortical plots
# def _canonicalize_aseg_for_ggseg(name: str) -> str | None:
#     """
#     Map your feature names to ggseg's expected FreeSurfer aseg labels.
#     Returns None if we decide to drop a region (very uncommon).
#     """
#     # Minor synonyms commonly encountered
#     syn = {
#         # FS sometimes uses 'Thalamus-Proper'
#         "Left-Thalamus": "Left-Thalamus-Proper",
#         "Right-Thalamus": "Right-Thalamus-Proper",
#         # Some users store 'Accumbens area' vs 'Accumbens-area'
#         "Left-Accumbens area": "Left-Accumbens-area",
#         "Right-Accumbens area": "Right-Accumbens-area",
#     }
#     return syn.get(name, name)
#
#
# # Regions ggseg almost always supports (fallback if we hit an exception)
# _SAFE_ASEG_CORE = {
#     # ventricles & brain stem
#     "3rd-Ventricle", "4th-Ventricle", "Brain-Stem",
#     "Left-Lateral-Ventricle", "Right-Lateral-Ventricle",
#     "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent",
#     # cerebellum
#     "Left-Cerebellum-White-Matter", "Right-Cerebellum-White-Matter",
#     "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex",
#     # limbic/basal ganglia set (classic 16)
#     "Left-Accumbens-area", "Right-Accumbens-area",
#     "Left-Amygdala", "Right-Amygdala",
#     "Left-Caudate", "Right-Caudate",
#     "Left-Hippocampus", "Right-Hippocampus",
#     "Left-Pallidum", "Right-Pallidum",
#     "Left-Putamen", "Right-Putamen",
#     "Left-Thalamus-Proper", "Right-Thalamus-Proper",
# }
#
# def _subcort_series_from_shap(exp: shap.Explanation, subcort_names: list[str]) -> pd.Series:
#     mas = mean_abs_shap_by_feature(exp)
#     keep = [n for n in subcort_names if n in mas.index]
#     s = mas.loc[keep].copy()
#     # normalize names for ggseg
#     s.index = [(_canonicalize_aseg_for_ggseg(n) or n) for n in s.index]
#     return s.sort_index()
#
# def plot_subcortical_ggseg_from_shap(exp: shap.Explanation,
#                                      subcort_feature_names: list[str],
#                                      title: str,
#                                      out_png: Path,
#                                      cmap: str = "Spectral",
#                                      background: str = "k",
#                                      edgecolor: str = "w",
#                                      bordercolor: str = "gray",
#                                      ylabel: str = "Mean |SHAP|",
#                                      dpi: int = 300) -> None:
#     """
#     Build a data dict for ggseg.plot_aseg from SHAP values and save a PNG.
#     Falls back to a 'safe core' label set if ggseg raises due to unknown labels.
#     """
#     try:
#         import ggseg
#     except Exception as e:
#         raise ImportError("python-ggseg is required. Install with: pip install ggseg") from e
#
#     import matplotlib.pyplot as plt
#
#     s = _subcort_series_from_shap(exp, subcort_feature_names)
#     if s.empty:
#         print("[skip] ggseg subcort: no subcortical features present in this SHAP object.")
#         return
#
#     data = {k: float(v) for k, v in s.items()}  # ggseg expects a mapping
#
#     def _try_plot(d):
#         fig = ggseg.plot_aseg(
#             d,
#             cmap=cmap,
#             background=background,
#             edgecolor=edgecolor,
#             bordercolor=bordercolor,
#             ylabel=ylabel,
#             title=title,
#         )
#         # Some ggseg versions return fig, others set current fig. Be robust:
#         if fig is None:
#             fig = plt.gcf()
#         return fig
#
#     # First attempt with all regions we have
#     try:
#         fig = _try_plot(data)
#     except Exception as e:
#         # Fallback: filter to the safe core set
#         kept = {k: v for k, v in data.items() if k in _SAFE_ASEG_CORE}
#         dropped = sorted(set(data) - set(kept))
#         print(f"[warn] ggseg error: {e}\n       Retrying with a reduced label set "
#               f"({len(kept)}/{len(data)} kept). Dropped: {', '.join(dropped[:12])}"
#               f"{' ...' if len(dropped) > 12 else ''}")
#         if not kept:
#             print("[skip] ggseg subcort: nothing left after filtering; skipping.")
#             return
#         fig = _try_plot(kept)
#
#     fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
#     plt.close(fig)
#     print(f"[ok] ggseg subcort saved → {out_png}")
#
# # ===========================
# # Batch: ggseg subcort 2D map
# # ===========================
# GGSEG_OUT_ROOT = OUT_DIR / "subcortical_ggseg"
# GGSEG_OUT_ROOT.mkdir(parents=True, exist_ok=True)
#
# for cohort in COHORTS:
#     pkl = shap_pickle_path(cohort)
#     if not pkl.exists():
#         print(f"[skip] Missing SHAP for {cohort}: {pkl}")
#         continue
#
#     exp = load_explanation(pkl)
#     pretty_site = SITE_LABEL.get(cohort, cohort)
#     concrete_t = concrete_target_for_site(cohort, TARGET)
#     title = f"{pretty_site} – {pretty_target_label(concrete_t)} (Subcortex • ggseg)"
#
#     out_png = GGSEG_OUT_ROOT / f"{MODEL}_{TARGET}_{cohort}_Subcortex_ggseg.png"
#     plot_subcortical_ggseg_from_shap(
#         exp=exp,
#         subcort_feature_names=Subcortical,  # from your feature_lists.pkl
#         title=title,
#         out_png=out_png,
#         cmap="Spectral",
#         background="k",
#         edgecolor="w",
#         bordercolor="gray",
#         ylabel="Mean |SHAP|",
#         dpi=300,
#     )
