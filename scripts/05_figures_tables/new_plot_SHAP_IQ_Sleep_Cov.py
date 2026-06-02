import os
import sys
from pathlib import Path
from typing import Optional

# project utils (adjust if needed)
sys.path.append('/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/lib/')
import utils  # noqa

import pickle
import numpy as np
import pandas as pd

import shap
import shapiq
from shapiq.interaction_values import aggregate_interaction_values

import umap.umap_ as umap
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import collections, collections.abc, six, sklearn
collections.Iterable = collections.abc.Iterable
sklearn.externals.six = six
from skrules import SkopeRules
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Matplotlib / Seaborn style
sns.set_context("paper")
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})

# %%
# ===========
# Constants
# ===========
FS_TICK  = 12
FS_LABEL = 12
FS_TITLE = 14

DEFAULT_FIGSIZE_SQ = (3.6, 3.0)  # 6, 5; 5.4, 4.5; 4.8, 4.0
DEFAULT_FIGSIZE_RX = (6, 4)  # 6, 4

# Set this to True to show figures interactively (PyCharm SciView); False to only save.
SHOW_PLOTS = False  # True of False

LABEL_REPLACEMENTS = {
    "Age_at_Scan": "Age",
    "Self_Sleep_Dur": "Self Sleep Duration",
    "Self_Sleep_Eff": "Self Sleep Efficiency",
    "PSG_Sleep_Dur": "PSG Sleep Duration",
    "PSG_Sleep_Eff": "PSG Sleep Efficiency",
    "Depression_score": "Depressive Score",
}

BASE_COLORS = [
    "#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
    "#984ea3", "#999999", "#e41a1c", "#dede00"
]

# %%
# ============================================
# Helpers: labels, styling, save/show behavior
# ============================================
def pretty_names(names: list[str]) -> list[str]:
    return [LABEL_REPLACEMENTS.get(n, n) for n in names]

def style_axes(ax):
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=FS_TITLE)
    return ax

def maybe_save_show(outpath: Optional[str], show: bool = SHOW_PLOTS, dpi: int = 300, close: bool = True):
    fig = plt.gcf()

    if outpath:
        p = Path(outpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        suf = p.suffix.lower()

        if suf == ".png":
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        elif suf == ".svg":
            fig.savefig(p, bbox_inches="tight")
        elif suf == "":
            fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
            fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        else:
            fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show or not outpath:
        plt.show()
    if close:
        plt.close()

def cluster_palette(labels) -> dict:
    uniq = np.unique(labels)
    from itertools import cycle, islice
    return dict(zip(uniq, list(islice(cycle(BASE_COLORS), len(uniq)))))

# ==========================
# Data & preprocessing utils
# ==========================
def get_feature_blocks(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = df.columns.tolist()
    span = lambda a, b: cols[cols.index(a): cols.index(b) + 1]
    return dict(
        Thickness_DK       = span('lh_bankssts_thickness',  'rh_insula_thickness'),
        Thickness_Schaefer = span('LH_Vis_1_thickness',     'RH_Default_pCunPCC_9_thickness'),
        Area_DK            = span('lh_bankssts_area',       'rh_insula_area'),
        Area_Schaefer      = span('LH_Vis_1_area',          'RH_Default_pCunPCC_9_area'),
        Subcortical        = span('Left-Lateral-Ventricle', 'CC_Anterior'),
    )

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Unit conversions, brain-size normalization, dtype fixes."""
    sleep_dur_cols = ['PSG_Sleep_Dur', 'Self_Sleep_Dur']
    sleep_eff_cols = ['PSG_Sleep_Eff', 'Self_Sleep_Eff']
    df_ = utils.convert_units(df.copy(), sleep_dur_cols, sleep_eff_cols)

    # Memory to accuracy (0–100)
    df_['Memory_Test'] = df_['Memory_Test'].apply(lambda x: x / 16) * 100

    blocks = get_feature_blocks(df_)
    for block in ('Thickness_DK', 'Thickness_Schaefer', 'Area_DK', 'Area_Schaefer'):
        df_[blocks[block]] = df_[blocks[block]].div(df_[blocks[block]].sum(axis=1), axis=0)
    df_[blocks['Subcortical']] = df_[blocks['Subcortical']].div(df_['EstimatedTotalIntraCranialVol'], axis=0)

    df_[['SEX', 'APOE4']] = df_[['SEX', 'APOE4']].astype('category')
    return df_

def feature_lists(df: pd.DataFrame) -> dict[str, list[str]]:
    blocks = get_feature_blocks(df)
    Sleep = ['PSG_Sleep_Dur', 'PSG_Sleep_Eff', 'Self_Sleep_Dur', 'Self_Sleep_Eff', 'Depression_score']
    Cov   = ['Age_at_Scan', 'SEX', 'BMI']
    Brain = blocks['Thickness_DK'] + blocks['Thickness_Schaefer'] + blocks['Area_DK'] + blocks['Area_Schaefer'] + blocks['Subcortical']
    return dict(Sleep=Sleep, Cov=Cov, Brain=Brain)

# ==========================
# Loading helpers (paths dict provided by driver)
# ==========================
def select_target_column(target: str) -> str:
    mapping = {"Stroop": "Stroop_Test", "Memory": "Memory_Test"}
    if target not in mapping:
        raise ValueError("target must be 'Stroop' or 'Memory'")
    return mapping[target]

def load_explanations(paths: dict) -> tuple[shap.Explanation, list]:
    with open(paths['iq_dir']   + 'ivs_SHIP_SHIP.pkl', 'rb') as f:
        ivs = pickle.load(f)
    with open(paths['shap_dir'] + 'explanation_train_set_train.pkl', 'rb') as f:
        explanation = pickle.load(f)
    return explanation, ivs

def shap_df_from_explanation(explanation: shap.Explanation) -> pd.DataFrame:
    if hasattr(explanation, "values") and hasattr(explanation, "feature_names"):
        return pd.DataFrame(explanation.values, columns=list(explanation.feature_names))
    if isinstance(explanation, list) and len(explanation) > 0:
        e0 = explanation[0]
        return pd.DataFrame(e0.values, columns=list(e0.feature_names))
    raise TypeError("Unsupported explanation type for SHAP DataFrame extraction")

# ==========================
# Clustering & embedding
# ==========================
def decide_k(target: str) -> int:
    return 4 if "Stroop" in target else 6

def spectral_cluster_and_umap(X_embed: np.ndarray, k: int,
                              umap_params: Optional[dict] = None):
    spec = SpectralClustering(
        n_clusters=k, affinity="nearest_neighbors",
        n_neighbors=15, assign_labels="kmeans", random_state=42
    )
    labels = spec.fit_predict(X_embed).astype(str)
    umap_params = umap_params or dict(
        n_components=2, n_neighbors=40, min_dist=0.1,
        metric="euclidean", random_state=42
    )
    emb = umap.UMAP(**umap_params).fit_transform(X_embed)
    return labels, emb

def reps_by_mean_abs_shap(shap_df: pd.DataFrame, labels) -> dict:
    reps = {}
    for cl in np.unique(labels):
        if cl == "-1":
            continue
        idx = np.where(labels == cl)[0]
        scores = shap_df.iloc[idx].abs().mean(axis=1)
        reps[cl] = scores.idxmax()
    return reps

def aggregate_iv_for_subset(ivs: list, subset_idx: np.ndarray):
    iv_list = [ivs[i] for i in subset_idx]
    return aggregate_interaction_values(iv_list, aggregation="mean")

# ==========================
# Plot helpers
# ==========================
def plot_umap(df_umap: pd.DataFrame, labels_sorted, fig_size=DEFAULT_FIGSIZE_RX, out=None, show=SHOW_PLOTS):
    plt.figure(figsize=fig_size)
    ax = sns.scatterplot(
        data=df_umap, x="U1", y="U2",
        hue="Cluster",
        palette=cluster_palette(df_umap["Cluster"].values),
        hue_order=labels_sorted,
        s=50, alpha=0.9, linewidth=0
    )
    ax.set_title("UMAP – Spectral clustering", fontsize=FS_TITLE)
    ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
    ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(title="Cluster", frameon=False, bbox_to_anchor=(1.00, 1), loc="upper left")  # bbox_to_anchor=(1.05, 1)
    plt.tight_layout()
    maybe_save_show(out, show)

def plot_feature_vs_shap(df_feat_shap: pd.DataFrame, labels_sorted, feat: str,
                         fig_size=DEFAULT_FIGSIZE_RX, out=None, show=SHOW_PLOTS):
    plt.figure(figsize=fig_size)
    ax = sns.scatterplot(
        data=df_feat_shap, x="Original", y="SHAP",
        hue="Cluster",
        palette=cluster_palette(df_feat_shap["Cluster"].values),
        hue_order=labels_sorted, s=50, linewidth=0
    )
    ax.set_xlabel(f"{LABEL_REPLACEMENTS.get(feat, feat)} (Original)", fontsize=FS_LABEL)
    ax.set_ylabel("SHAP value", fontsize=FS_LABEL)
    ax.set_title(f"{LABEL_REPLACEMENTS.get(feat, feat)}: Original vs SHAP", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(title="Cluster", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    maybe_save_show(out, show)

# def beeswarm_cluster(explanation: shap.Explanation, labels, cl, feature_names: list[str],
#                      fig_size=DEFAULT_FIGSIZE_SQ, out=None, show=SHOW_PLOTS):
#     plt.figure(figsize=fig_size)
#     shap.plots.beeswarm(
#         explanation[labels == cl],
#         clustering=False, show=False,
#         # feature_names=pretty_names(feature_names)
#     )
#     plt.title(f"Beeswarm – Cluster {cl}", fontsize=FS_TITLE)
#     plt.tight_layout()
#     maybe_save_show(out, show)
def beeswarm_cluster(explanation: shap.Explanation,
                     labels,
                     cl,
                     feature_names: list[str] | None = None,
                     fig_size=DEFAULT_FIGSIZE_RX,
                     out=None,
                     show=SHOW_PLOTS,
                     max_display: int = 20,
                     sort: bool = True):
    """
    Cluster-specific SHAP summary (dot) plot.
    Uses the cluster-sliced explanation's own feature order to avoid mislabeling.
    """
    exp_cl = explanation[labels == cl]
    pretty = [LABEL_REPLACEMENTS.get(n, n) for n in exp_cl.feature_names]

    shap.summary_plot(
        exp_cl,
        features=exp_cl.data,
        feature_names=pretty,
        plot_type="dot",
        sort=sort,
        show=False,
        plot_size=(fig_size[0], fig_size[1]) if fig_size else (6, 6),
        max_display=max_display,
    )

    ax = plt.gca()
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    ax.set_title(f"Beeswarm – Cluster {cl}", fontsize=FS_TITLE)
    plt.tight_layout()
    maybe_save_show(out, show)

def umap_colored_by_shap(emb: np.ndarray, vals: np.ndarray, feat_label: str,
                         vmin=-4, vmax=4, fig_size=DEFAULT_FIGSIZE_RX, out=None, show=SHOW_PLOTS):
    plt.figure(figsize=fig_size)
    sc = plt.scatter(emb[:, 0], emb[:, 1],
                     c=np.clip(vals, vmin, vmax),
                     cmap="coolwarm", vmin=vmin, vmax=vmax,
                     s=50, linewidths=0)
    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_label(f"{feat_label} SHAP", rotation=270, labelpad=15)
    plt.xlabel("UMAP 1", fontsize=FS_LABEL)
    plt.ylabel("UMAP 2", fontsize=FS_LABEL)
    plt.title(f"UMAP colored by {feat_label} SHAP", fontsize=FS_TITLE)
    plt.tick_params(labelsize=FS_TICK)
    plt.tight_layout()
    maybe_save_show(out, show)

# def umap_shap_grid(
#     emb: np.ndarray,
#     shap_df: pd.DataFrame,
#     rows: list[list[str]],
#     *,
#     fig_size: tuple[float, float] = (7.0, 8.0),
#     cmap: str = "RdBu_r",
#     vmin: float = -4.0,
#     vmax: float = 4.0,
#     out: str | None = None,
#     show: bool = SHOW_PLOTS,
# ) -> None:
#     base_df = pd.DataFrame({"U1": emb[:, 0], "U2": emb[:, 1]})
#
#     nrows = len(rows)
#     ncols = max(len(r) for r in rows)
#
#     fig, axes = plt.subplots(
#         nrows=nrows,
#         ncols=ncols,
#         figsize=fig_size,
#         sharex=True,
#         sharey=True,
#     )
#
#     # Normalize axes shape
#     if nrows == 1:
#         axes = np.array([axes])
#     if ncols == 1:
#         axes = axes.reshape(nrows, 1)
#
#     last_sc = None
#
#     for r, row in enumerate(rows):
#         for c in range(ncols):
#             ax = axes[r, c]
#             if c >= len(row):
#                 ax.set_visible(False)
#                 continue
#
#             feat = row[c]
#             if feat not in shap_df.columns:
#                 ax.set_visible(False)
#                 continue
#
#             vals = np.clip(shap_df[feat].values, vmin, vmax)
#             last_sc = ax.scatter(
#                 base_df["U1"], base_df["U2"],
#                 c=vals, cmap=cmap,
#                 vmin=vmin, vmax=vmax,
#                 s=35, linewidths=0,
#             )
#             ax.set_title(LABEL_REPLACEMENTS.get(feat, feat),
#                          fontsize=FS_TITLE-3)
#             ax.tick_params(length=3, labelsize=FS_TICK-2)
#
#     # ---- layout: reserve left+bottom for labels, right for colourbar ----
#     # rect = [left, bottom, right, top] in figure coordinates
#     fig.tight_layout(rect=[0.16, 0.16, 0.88, 0.98])
#
#     # Colourbar in its own axis on the right (no overlap with subplots)
#     if last_sc is not None:
#         cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
#         cbar = fig.colorbar(last_sc, cax=cax)
#         cbar.set_ticks([vmin, 0, vmax])
#         cbar.set_label("SHAP value", rotation=270, labelpad=15,
#                        fontsize=FS_LABEL)
#         cbar.ax.tick_params(labelsize=FS_TICK-1)
#
#     # Place super labels inside the reserved margins (below/left of axes)
#     fig.supxlabel("UMAP 1", y=0.16, fontsize=FS_LABEL)
#     fig.supylabel("UMAP 2", x=0.16, fontsize=FS_LABEL)
#
#     maybe_save_show(out, show)
def umap_shap_grid(
    emb: np.ndarray,
    shap_df: pd.DataFrame,
    rows: list[list[str]],
    *,
    fig_size: tuple[float, float] = (7.0, 8.0),
    cmap: str = "RdBu_r",
    vmin: float = -4.0,
    vmax: float = 4.0,
    out: str | None = None,
    show: bool = SHOW_PLOTS,
) -> None:
    """
    Grid of UMAP projections coloured by SHAP value for multiple features.

    Parameters
    ----------
    emb       : (n_samples, 2) UMAP embedding
    shap_df   : DataFrame with SHAP values (same row order as emb)
    rows      : nested list of feature names, e.g.
                [
                    ["Age_at_Scan", "BMI"],
                    ["PSG_Sleep_Dur", "PSG_Sleep_Eff"],
                    ["Self_Sleep_Dur", "Self_Sleep_Eff"],
                ]
    """
    base_df = pd.DataFrame({"U1": emb[:, 0], "U2": emb[:, 1]})

    nrows = len(rows)
    ncols = max(len(r) for r in rows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=fig_size,
        sharex=True,
        sharey=True,
    )

    # Normalize axes shape
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    last_sc = None

    for r, row in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(row):
                ax.set_visible(False)
                continue

            feat = row[c]
            if feat not in shap_df.columns:
                ax.set_visible(False)
                continue

            vals = np.clip(shap_df[feat].values, vmin, vmax)
            last_sc = ax.scatter(
                base_df["U1"], base_df["U2"],
                c=vals,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=35,
                linewidths=0,
            )
            ax.set_title(LABEL_REPLACEMENTS.get(feat, feat),
                         fontsize=FS_TITLE - 3)
            ax.tick_params(length=3, labelsize=FS_TICK - 2)

    # ---- layout: reserve left+bottom for labels, right for colourbar ----
    # rect = [left, bottom, right, top] in figure coordinates
    axes_rect = [0.16, 0.18, 0.86, 0.96]
    fig.tight_layout(rect=axes_rect)

    # Colourbar in its own axis on the right (aligned with subplot block)
    if last_sc is not None:
        # keep vertical centre but shorten the bar a lot
        cb_height = 0.30                      # 0.50
        cb_center = (axes_rect[1] + axes_rect[3]) / 2.0  # ~0.57
        cb_bottom = cb_center - cb_height / 2.0          # ~0.32

        cb_left  = axes_rect[2] + 0.02        # to the right of subplots
        cb_width = 0.02

        cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])

        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_label("SHAP value", rotation=270, labelpad=15,
                       fontsize=FS_LABEL)
        cbar.ax.tick_params(labelsize=FS_TICK - 1)

    # Super labels: centred, placed closer to subplot region
    # fig.supxlabel("UMAP 1", y=axes_rect[1] - 0.00, fontsize=FS_LABEL)  # - 0.01
    # fig.supylabel("UMAP 2", x=axes_rect[0] - 0.00, fontsize=FS_LABEL)  # - 0.01
    x_center = (axes_rect[0] + axes_rect[2]) / 2.0
    y_center = (axes_rect[1] + axes_rect[3]) / 2.0
    fig.supxlabel("UMAP 1", y=axes_rect[1] - 0.00, x=x_center, fontsize=FS_LABEL)
    fig.supylabel("UMAP 2", x=axes_rect[0] - 0.00, y=y_center, fontsize=FS_LABEL)

    maybe_save_show(out, show)


# def plot_si_graph(iv_obj, feature_names: list[str], title: str,
#                   fig_size=DEFAULT_FIGSIZE_RX, out=None, show=SHOW_PLOTS):
#     plt.figure(figsize=fig_size)
#     fig, ax = shapiq.si_graph_plot(
#         interaction_values=iv_obj,
#         feature_names=feature_names,
#         show=False,
#         min_max_order=(1, 2),
#         size_factor=5.0,
#     )
#     plt.title(title, fontsize=FS_TITLE, pad=10)
#     plt.tight_layout()
#     # Prefer saving the figure handle returned by shapiq
#     if out:
#         Path(out).parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(out, dpi=300, bbox_inches="tight")
#     if show or not out:
#         plt.show()
#     plt.close(fig)
def plot_si_graph(iv_obj, feature_names: list[str], title: str,
                  fig_size=DEFAULT_FIGSIZE_SQ, out=None, show=SHOW_PLOTS,
                  size_factor: float = 1.0, min_max_interactions: tuple | None = (-1, 1)):
    """SHAP-IQ network plot with controllable figure size and pretty labels."""
    pretty = [LABEL_REPLACEMENTS.get(n, n) for n in feature_names]
    fig, ax = shapiq.si_graph_plot(
        interaction_values=iv_obj,
        feature_names=pretty,
        show=False,
        min_max_order=(1, 2),
        size_factor=size_factor,
        min_max_interactions=min_max_interactions,
    )
    if fig_size is not None:
        fig.set_size_inches(fig_size[0], fig_size[1], forward=True)

    for txt in ax.texts:
        txt.set_fontsize(10)

    # ax.set_title(title, fontsize=FS_TITLE)
    ax.set_title(title, fontsize=FS_TITLE-2, pad=20)  #title, fontsize=FS_TITLE, pad=20
    # plt.title(title, fontsize=FS_TITLE)
    fig.tight_layout()
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        suf = p.suffix.lower()

        if suf == ".png":
            fig.savefig(p, dpi=300, bbox_inches="tight")
            fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        elif suf == ".svg":
            fig.savefig(p, bbox_inches="tight")
        elif suf == "":
            fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
            fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        else:
            fig.savefig(p, dpi=300, bbox_inches="tight")

# ==========================
# Cluster feature summary
# ==========================
def cluster_feature_summary(df: pd.DataFrame, labels, categorical_cols=None,
                            decimals=2, out_csv: Optional[str] = None) -> pd.DataFrame:
    df_ = df.copy()
    df_["Cluster"] = labels
    clusters = sorted(df_["Cluster"].unique())
    g = df_.groupby("Cluster")

    numeric_cols = df_.select_dtypes(include=[np.number]).columns
    rows, idx = [], []

    for col in numeric_cols:
        means = g[col].mean(); stds = g[col].std()
        mins  = g[col].min();  maxs = g[col].max()
        rows.append([f"{means[c]:.{decimals}f} ± {stds[c]:.{decimals}f}" for c in clusters]); idx.append((col, "Mean ± SD"))
        rows.append([f"{mins[c]:.{decimals}f}–{maxs[c]:.{decimals}f}" for c in clusters]);   idx.append((col, "Range"))

    if categorical_cols:
        for col in categorical_cols:
            counts = g[col].value_counts().unstack(fill_value=0).sort_index(axis=1)
            for level in counts.columns:
                rows.append([int(counts.loc[c, level]) for c in clusters])
                idx.append((col, f"{level}"))

    out_df = pd.DataFrame(
        rows,
        columns=[f"Cluster {c} (N={g.size()[c]})" for c in clusters],
        index=pd.MultiIndex.from_tuples(idx, names=["Feature", "Statistic"])
    )
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv)
    return out_df

# ==========================
# SkopeRules per cluster (optional)
# ==========================
def fit_skope_by_cluster(
        df_X: pd.DataFrame,
        y_clust: pd.Series,
        *,
        n_estimators: int = 30,
        recall_min:   float = 0.30,
        max_depth:    int = 4,
        max_depth_duplication: int = 6,
        max_samples:  float = 0.80,
        random_state: int = 42
) -> OrderedDict[str, SkopeRules]:
    """Train one SkopeRules model per cluster; stable ordering."""
    models: OrderedDict[str, SkopeRules] = OrderedDict()
    for cl in sorted(pd.Series(y_clust).unique(), key=str):
        y_bin = (y_clust == cl).astype(int).values  # 1 = in cluster cl
        sr = SkopeRules(
            feature_names=df_X.columns.tolist(),
            n_estimators=n_estimators,
            recall_min=recall_min,
            max_depth=max_depth,
            max_depth_duplication=max_depth_duplication,
            max_samples=max_samples,
            max_features=None,  # try all features
            random_state=random_state
        ).fit(df_X, y_bin)
        models[str(cl)] = sr
    return models

def print_top_rules(sr_models: dict[str, SkopeRules]) -> None:
    for cl, sr in sr_models.items():
        print(f"\n=== Cluster {cl} ===")
        if not getattr(sr, "rules_", []):
            print("No rule met precision/recall thresholds.")
            continue
        rule_str, (prec, rec, _) = sr.rules_[0]
        pretty = " AND ".join(rule_str.split(" and "))
        print(f"IF {pretty}")
        print(f"→ precision = {prec:.3f}, recall (coverage) = {rec:.3f}")

def print_iv_extremes(iv_obj, feature_names: list[str], prefix: str = "") -> None:
    """
    Print largest/smallest (signed) interactions for 1st- and 2nd-order.
    Uses pretty labels. For 2nd-order, searches the upper triangle (i<j).
    """
    pn = [LABEL_REPLACEMENTS.get(n, n) for n in feature_names]

    # --- 1st-order (shape: F,) ---
    first = iv_obj.get_n_order_values(1)  # main effects
    i_max = int(np.argmax(first));  v_max = float(first[i_max])
    i_min = int(np.argmin(first));  v_min = float(first[i_min])
    print(f"{prefix} 1st-order max: {pn[i_max]} = {v_max:.6f}")
    print(f"{prefix} 1st-order min: {pn[i_min]} = {v_min:.6f}")

    # --- 2nd-order (shape: F x F), use upper triangle only ---
    second = iv_obj.get_n_order_values(2)
    F = second.shape[0]
    iu, ju = np.triu_indices(F, k=1)
    vals = second[iu, ju]

    k_max = int(np.argmax(vals)); k_min = int(np.argmin(vals))
    imax, jmax = iu[k_max], ju[k_max]; vmax2 = float(second[imax, jmax])
    imin, jmin = iu[k_min], ju[k_min]; vmin2 = float(second[imin, jmin])

    print(f"{prefix} 2nd-order max: ({pn[imax]}, {pn[jmax]}) = {vmax2:.6f}")
    print(f"{prefix} 2nd-order min: ({pn[imin]}, {pn[jmin]}) = {vmin2:.6f}")

    # (Optional) also report largest magnitude pair:
    k_abs = int(np.argmax(np.abs(vals)))
    ia, ja = iu[k_abs], ju[k_abs]
    vab = float(second[ia, ja])
    print(f"{prefix} 2nd-order max|abs|: ({pn[ia]}, {pn[ja]}) = {vab:.6f}")

# %%
# ==========================
# ===== Interactive driver
# ==========================
# Choose target for this session
target = "Stroop"   # "Stroop" or "Memory"

# IO roots
data_root    = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
results_root = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/models/AutoGluon/'

# Preferred figure sizes (tweak as needed)
fig_size_sq = DEFAULT_FIGSIZE_SQ
fig_size_rx = DEFAULT_FIGSIZE_RX

print(f"\nStarting SHAP-IQ (Sleep_Cov) → target={target}\n")

# --- build paths here (no helper function) ---
_ = select_target_column(target)  # validates target
case_root = f"{results_root}{target}/Sleep_Cov/"
paths = {
    "data_csv": f"{data_root}SHIP_Trend_dataset_renamed.csv",
    "case_root": case_root,
    "shap_dir": f"{case_root}SHAP/",
    "iq_dir": f"{case_root}shapiq/",
    "cluster": f"{case_root}SHAP_clustering/",
}

# %%
# Ensure dirs exist
os.makedirs(paths['cluster'], exist_ok=True)
os.makedirs(paths['cluster'] + "spectral_final_scatter/", exist_ok=True)
os.makedirs(paths['cluster'] + "umap_by_feature/", exist_ok=True)
os.makedirs(paths['cluster'] + "rep_networks/", exist_ok=True)
os.makedirs(paths['cluster'] + "cluster_avg_networks/", exist_ok=True)

# --- data ---
df = pd.read_csv(paths['data_csv']).dropna(subset=['Stroop_Test', 'Memory_Test'])
df_ml = preprocess(df)
feats = feature_lists(df_ml)
X = feats['Sleep'] + feats['Cov']          # Sleep_Cov only
feature_names = X

# --- load explanations ---
explanation, ivs = load_explanations(paths)
shap_df = shap_df_from_explanation(explanation)

# --- cluster + embed ---
k = decide_k(target)
labels, emb = spectral_cluster_and_umap(shap_df.values, k)
labels_sorted = sorted(np.unique(labels))
df_umap = pd.DataFrame({"U1": emb[:, 0], "U2": emb[:, 1], "Cluster": labels})

# --- 1) UMAP overview ---
plot_umap(df_umap, labels_sorted, fig_size=fig_size_rx,
          out=f"{paths['cluster']}spectral_final_k{k}_umap.png", show=SHOW_PLOTS)

# --- 2) Feature↔SHAP scatter (key features) ---
for feat in ["Age_at_Scan", "PSG_Sleep_Dur", "Self_Sleep_Dur", "PSG_Sleep_Eff", "Self_Sleep_Eff"]:
    if feat in df_ml.columns and feat in shap_df.columns:
        dfp = pd.DataFrame({
            "Original": df_ml[feat].values,
            "SHAP": shap_df[feat].values,
            "Cluster": labels
        })
        plot_feature_vs_shap(
            dfp, labels_sorted, feat,
            fig_size=fig_size_rx,
            out=f"{paths['cluster']}spectral_final_scatter/{feat}_spectral_k{k}.png",
            show=SHOW_PLOTS
        )

# --- 3) Beeswarm per cluster (pretty names) ---
for cl in labels_sorted:
    beeswarm_cluster(
        explanation, labels, cl, feature_names,
        fig_size=fig_size_rx,
        out=f"{paths['cluster']}beeswarm_cluster_{cl}.png",
        show=SHOW_PLOTS
    )

# --- 4) UMAP colored by SHAP (selected features) ---
for feat in ["PSG_Sleep_Dur", "Self_Sleep_Dur", "PSG_Sleep_Eff", "Self_Sleep_Eff", "Age_at_Scan", "BMI"]:
    if feat in shap_df.columns:
        umap_colored_by_shap(
            emb, shap_df[feat].values, LABEL_REPLACEMENTS.get(feat, feat),
            fig_size=fig_size_rx,
            out=f"{paths['cluster']}umap_by_feature/{feat}_SHAP_on_UMAP.png",
            show=SHOW_PLOTS
        )

# --- UMAP SHAP grid ---
rows = [
    ["Age_at_Scan", "BMI"],
    ["PSG_Sleep_Dur", "PSG_Sleep_Eff"],
    ["Self_Sleep_Dur", "Self_Sleep_Eff"],
]

umap_shap_grid(
    emb,
    shap_df,
    rows=rows,
    fig_size=(7.0, 9.5),  # fig_size=(7.0, 8.0)
    cmap="RdBu_r",
    vmin=-4,
    vmax=4,
    out=f"{paths['cluster']}umap_SHAP_grid.png",
    show=SHOW_PLOTS,
)

# --- 5) Representative subjects (max mean |SHAP|) networks ---
reps = reps_by_mean_abs_shap(shap_df, labels)
for cl, idx in reps.items():
    plot_si_graph(
        ivs[idx], feature_names,
        title=f"Cluster {cl} – representative subject",
        fig_size=fig_size_sq,
        out=f"{paths['cluster']}rep_networks/cluster_{cl}_rep_network.png",
        show=SHOW_PLOTS
    )

# --- 6) Cluster-mean interaction networks + beeswarm + SHAP-IQ bars ---
for cl in [c for c in labels_sorted if c != "-1"]:
    idx = np.where(labels == cl)[0]
    if len(idx) == 0:
        continue

    iv_avg = aggregate_iv_for_subset(ivs, idx)

    print_iv_extremes(iv_avg, feature_names, prefix=f"[Cluster {cl}]")

    plot_si_graph(
        iv_avg, feature_names,
        title=f"Cluster {cl} – mean interaction network (n={len(idx)})",
        fig_size=fig_size_sq,
        out=f"{paths['cluster']}cluster_avg_networks/cluster_{cl}_avg_network.png",
        show=SHOW_PLOTS
    )

    beeswarm_cluster(
        explanation, labels, cl, feature_names,
        fig_size=fig_size_rx,
        out=f"{paths['cluster']}cluster_avg_networks/beeswarm_cluster_{cl}.png",
        show=SHOW_PLOTS
    )

    plt.figure(figsize=fig_size_rx)
    shapiq.plot.bar_plot([ivs[i] for i in idx],
                         feature_names=pretty_names(feature_names),
                         abbreviate=False, max_display=20, show=False)
    plt.title(f"Cluster {cl} – SHAP-IQ top-20", fontsize=FS_TITLE)
    plt.tight_layout()
    maybe_save_show(f"{paths['cluster']}cluster_avg_networks/cluster_{cl}_bar.png", SHOW_PLOTS)

# --- 7) All-subject mean interaction network ---
iv_all_avg = aggregate_interaction_values(ivs, aggregation="mean")
plot_si_graph(
    iv_all_avg, feature_names,
    title="Mean 1st/2nd order interactions (all subjects)",
    fig_size=fig_size_sq,
    out=f"{paths['cluster']}network_mean_all_subjects.png",
    show=SHOW_PLOTS
)

# --- 8) Cluster feature summary CSV ---
summary = cluster_feature_summary(
    df=df_ml[X], labels=labels, categorical_cols=["SEX"],
    out_csv=f"{paths['case_root']}cluster_feature_summary.csv"
)
print("✓ Saved cluster_feature_summary.csv:", f"{paths['case_root']}cluster_feature_summary.csv")
print("Summary shape:", summary.shape)

# --- 9) SkopeRules rules per cluster on original features (mandatory) ---
y_clust = pd.Series(labels, name="cluster").astype(str)
df_X = df_ml[X].reset_index(drop=True)

sr_models = fit_skope_by_cluster(df_X, y_clust)

for cl, sr in sr_models.items():
    print(f"\n=== Cluster {cl} ===")
    if not getattr(sr, "rules_", []):
        print("No rule met precision/recall thresholds.")
        continue
    rule_str, (prec, rec, _) = sr.rules_[0]   # “_” = count
    pretty = " AND ".join(rule_str.split(" and "))
    print(f"IF {pretty}")
    print(f"→ precision = {prec:.3f}, recall (coverage) = {rec:.3f}")