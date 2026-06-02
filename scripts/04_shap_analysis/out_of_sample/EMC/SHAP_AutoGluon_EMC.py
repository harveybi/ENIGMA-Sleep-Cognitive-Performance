"""
SHAP_subgroup_EMC.py

EMC-only subgroup analysis for feature combination: Sleep_Cov.

This script:
1. Loads EMC data.
2. Applies minimal preprocessing needed for Sleep_Cov.
3. Loads SHIP-derived subgroup rules from subgroup_artifacts_*.pkl.
4. Assigns EMC subjects to SHIP-derived subgroups.
5. Loads EMC SHAP and SHAP-IQ outputs.
6. Generates EMC site-level and subgroup-level plots.
7. Saves plot-ready derived artifacts without EMC raw/preprocessed feature values.

No SHIP raw data are required.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import shapiq
from shapiq.interaction_values import aggregate_interaction_values

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})
sns.set_context("paper")


# ==========================
# Fixed config
# ==========================
FEATURE_COMB = "Sleep_Cov"

TARGET_TO_COLUMN = {
    "Stroop": "Stroop_Test",
    "Memory": "Memory_Test",
    "Stroop_rgo_age": "Stroop_rgo_age",
    "Memory_rgo_age": "Memory_rgo_age",
}

SLEEP_DUR_COLS = ["PSG_Sleep_Dur", "Self_Sleep_Dur"]
SLEEP_EFF_COLS = ["PSG_Sleep_Eff", "Self_Sleep_Eff"]

LABEL_REPLACEMENTS = {
    "Age_at_Scan": "Age",
    "Self_Sleep_Dur": "Self Sleep Duration",
    "Self_Sleep_Eff": "Self Sleep Efficiency",
    "PSG_Sleep_Dur": "PSG Sleep Duration",
    "PSG_Sleep_Eff": "PSG Sleep Efficiency",
    "Depression_score": "Depressive Score",
}

FS_TICK = 12
FS_LABEL = 12
FS_TITLE = 14

DEFAULT_FIGSIZE_SQ = (3.6, 3.0)
DEFAULT_FIGSIZE_RX = (6.0, 4.0)


# ==========================
# Helper functions
# ==========================
def pretty_names(names, label_replacements=None):
    if label_replacements is None:
        label_replacements = LABEL_REPLACEMENTS
    return [label_replacements.get(n, n) for n in names]


def maybe_save_show(
    outpath: Optional[str],
    show: bool = False,
    dpi: int = 300,
    close: bool = True,
):
    fig = plt.gcf()

    if outpath:
        p = Path(outpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        suffix = p.suffix.lower()

        if suffix == ".png":
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        elif suffix == ".svg":
            fig.savefig(p, bbox_inches="tight")
        elif suffix == "":
            fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
            fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")
        else:
            fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show or not outpath:
        plt.show()

    if close:
        plt.close()


def validate_required_columns(df, cols, name="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name} is missing {len(missing)} required columns:\n"
            + "\n".join(missing)
        )


def make_rule_input(df, cols):
    """
    Convert rule input to numeric values while preserving original coding.

    Important:
    Do NOT use .cat.codes, because categorical values such as SEX 1/2 may become 0/1.
    """
    X_rule = df[cols].copy()

    for c in X_rule.columns:
        if str(X_rule[c].dtype) == "category":
            X_rule[c] = pd.to_numeric(X_rule[c].astype(str), errors="coerce")
        elif X_rule[c].dtype == "object":
            X_rule[c] = pd.to_numeric(X_rule[c], errors="coerce")

    return X_rule


def apply_subgroup_rules(df_site, *, rules, cols, unknown="unassigned"):
    """
    Apply exported SHIP-derived subgroup rules to EMC.
    """
    X_site = make_rule_input(df_site, cols)

    def _label(row):
        row_df = row.to_frame().T

        for cl, rule_list in rules.items():
            for rule in rule_list:
                try:
                    if row_df.query(rule).shape[0] > 0:
                        return str(cl)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to apply rule for cluster {cl}: {rule}"
                    ) from e

        return unknown

    labels = X_site.apply(_label, axis=1).astype(str)

    print("\nEMC subgroup counts:")
    print(labels.value_counts(dropna=False))

    return labels


def validate_row_alignment(df_site, labels, explanation, ivs):
    n_df = df_site.shape[0]
    n_labels = len(labels)
    n_shap = explanation.values.shape[0]
    n_shapiq = len(ivs)

    print("\nRow-count sanity check:")
    print(f"  EMC dataframe:     {n_df}")
    print(f"  subgroup labels:   {n_labels}")
    print(f"  SHAP explanation:  {n_shap}")
    print(f"  SHAP-IQ values:    {n_shapiq}")

    if not (n_df == n_labels == n_shap == n_shapiq):
        raise ValueError(
            "Row mismatch detected. EMC dataframe, subgroup labels, SHAP, and "
            "SHAP-IQ must have the same row order and sample size."
        )


def validate_feature_alignment(explanation, feature_cols):
    shap_values = np.asarray(explanation.values)

    if shap_values.shape[1] != len(feature_cols):
        raise ValueError(
            f"Feature mismatch: SHAP has {shap_values.shape[1]} columns, "
            f"but subgroup artifact has {len(feature_cols)} feature columns."
        )

    if hasattr(explanation, "feature_names") and explanation.feature_names is not None:
        exp_names = list(explanation.feature_names)

        if exp_names != list(feature_cols):
            print("\n[WARN] SHAP feature_names do not exactly match subgroup feature_cols.")
            print("       The script will continue using feature_cols from subgroup_artifacts.")
            print("       Please verify that SHAP/SHAP-IQ were generated using the same feature order.")


def preprocess_emc_sleep_cov(df, *, utils_module, feature_cols):
    """
    Minimal EMC preprocessing for Sleep_Cov only.

    No brain normalization.
    No shuffled features.
    No APOE-specific preprocessing.
    """
    df = df.copy()

    df = utils_module.convert_units(
        df,
        SLEEP_DUR_COLS,
        SLEEP_EFF_COLS,
    )

    # Keep original coding, but make categorical columns explicit where present.
    obj_cols = [c for c in ["SEX", "APOE4"] if c in df.columns]
    if obj_cols:
        df[obj_cols] = df[obj_cols].astype("category")

    # Convert numeric Sleep_Cov columns that exist.
    # Do not force SEX/APOE4 to float here because plotting can handle categories;
    # rule application converts them safely via make_rule_input().
    numeric_candidates = [
        "Age_at_Scan",
        "Depression_score",
        "BMI",
        "PSG_Sleep_Dur",
        "Self_Sleep_Dur",
        "PSG_Sleep_Eff",
        "Self_Sleep_Eff",
    ]

    numeric_cols = [
        c for c in numeric_candidates
        if c in df.columns and c in feature_cols
    ]

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].astype("float64")

    return df


def plot_si_graph(
    iv_obj,
    feature_names,
    title,
    out=None,
    show=False,
    fig_size=DEFAULT_FIGSIZE_SQ,
    size_factor=1.0,
    label_replacements=None,
):
    pretty_feats = pretty_names(feature_names, label_replacements)

    fig, ax = shapiq.si_graph_plot(
        interaction_values=iv_obj,
        feature_names=pretty_feats,
        show=False,
        min_max_order=(1, 2),
        size_factor=size_factor,
        min_max_interactions=None,
    )

    if fig_size is not None:
        fig.set_size_inches(fig_size[0], fig_size[1], forward=True)

    for txt in ax.texts:
        txt.set_fontsize(10)

    ax.set_title(title, fontsize=FS_TITLE - 2, pad=20)
    fig.tight_layout()

    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        fig.savefig(p.with_suffix(".svg"), bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def make_emc_site_plots(
    *,
    explanation,
    ivs,
    df_emc_ml,
    feature_names,
    save_dir,
    label_replacements=None,
    max_display=20,
    show=False,
):
    os.makedirs(save_dir, exist_ok=True)

    pretty_feats = pretty_names(feature_names, label_replacements)

    # 1. SHAP summary dot plot
    exp_pretty = pretty_names(list(explanation.feature_names), label_replacements)

    shap.summary_plot(
        explanation,
        features=explanation.data,
        feature_names=exp_pretty,
        plot_type="dot",
        sort=True,
        show=False,
        plot_size=DEFAULT_FIGSIZE_RX,
        max_display=max_display,
    )
    ax = plt.gca()
    ax.set_title("EMC – SHAP summary", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    plt.tight_layout()

    maybe_save_show(
        os.path.join(save_dir, "EMC_shap_summary.png"),
        show=show,
    )

    # 2. SHAP-IQ bar plot
    plt.figure(figsize=DEFAULT_FIGSIZE_RX)
    shapiq.plot.bar_plot(
        ivs,
        feature_names=pretty_feats,
        abbreviate=False,
        max_display=max_display,
        show=False,
    )
    ax = plt.gca()
    ax.set_title("EMC – SHAP-IQ bar", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    plt.tight_layout()

    maybe_save_show(
        os.path.join(save_dir, "EMC_shapiq_bar.png"),
        show=show,
    )

    # 3. SHAP-IQ beeswarm
    plt.figure(figsize=DEFAULT_FIGSIZE_RX)
    shapiq.plot.beeswarm_plot(
        ivs,
        df_emc_ml[feature_names],
        feature_names=pretty_feats,
        abbreviate=False,
        show=False,
    )
    ax = plt.gca()
    ax.set_title("EMC – SHAP-IQ beeswarm", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    plt.tight_layout()

    maybe_save_show(
        os.path.join(save_dir, "EMC_shapiq_beeswarm.png"),
        show=show,
    )

    print("[EMC] Site-level plots done.")


def make_emc_cluster_plots(
    *,
    labels,
    explanation,
    ivs,
    feature_names,
    save_dir,
    label_replacements=None,
    max_display=20,
    show=False,
):
    os.makedirs(save_dir, exist_ok=True)

    labels = np.asarray(labels).astype(str)
    unique_labels = sorted(np.unique(labels), key=str)
    pretty_feats = pretty_names(feature_names, label_replacements)

    for cl in unique_labels:
        idx = np.where(labels == cl)[0]

        if idx.size == 0:
            continue

        # 1. Mean SHAP-IQ interaction network
        iv_avg = aggregate_interaction_values(
            [ivs[i] for i in idx],
            aggregation="mean",
        )

        plot_si_graph(
            iv_avg,
            feature_names,
            title=f"EMC – Cluster {cl} (n={len(idx)})",
            out=os.path.join(save_dir, f"EMC_cl{cl}_network.png"),
            show=show,
            label_replacements=label_replacements,
        )

        # 2. Cluster SHAP summary
        exp_cl = explanation[idx]
        exp_pretty = pretty_names(list(exp_cl.feature_names), label_replacements)

        shap.summary_plot(
            exp_cl,
            features=exp_cl.data,
            feature_names=exp_pretty,
            plot_type="dot",
            sort=True,
            show=False,
            plot_size=DEFAULT_FIGSIZE_RX,
            max_display=max_display,
        )
        ax = plt.gca()
        ax.set_title(f"EMC – Cluster {cl} (n={len(idx)})", fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.label.set_size(FS_LABEL)
        ax.yaxis.label.set_size(FS_LABEL)
        plt.tight_layout()

        maybe_save_show(
            os.path.join(save_dir, f"EMC_cl{cl}_shap_summary.png"),
            show=show,
        )

        # 3. Cluster SHAP-IQ bar plot
        plt.figure(figsize=DEFAULT_FIGSIZE_RX)
        shapiq.plot.bar_plot(
            [ivs[i] for i in idx],
            feature_names=pretty_feats,
            max_display=max_display,
            abbreviate=False,
            show=False,
        )
        ax = plt.gca()
        ax.set_title(f"EMC – Cluster {cl} (n={len(idx)})", fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        ax.xaxis.label.set_size(FS_LABEL)
        ax.yaxis.label.set_size(FS_LABEL)
        plt.tight_layout()

        maybe_save_show(
            os.path.join(save_dir, f"EMC_cl{cl}_shapiq_bar.png"),
            show=show,
        )

        print(f"[EMC] Cluster {cl} plots done.")


def save_plot_ready_artifacts(
    *,
    out_dir,
    target,
    feature_comb,
    labels,
    explanation,
    ivs,
    feature_names,
    cluster_rules,
    rule_info,
    ship_cluster_counts,
    label_replacements=None,
    include_subject_level_values=False,
):
    """
    Save derived plotting artifacts.

    Default output does NOT include EMC raw/preprocessed feature values.
    """
    os.makedirs(out_dir, exist_ok=True)

    labels = pd.Series(labels).astype(str).reset_index(drop=True)
    feature_names = list(feature_names)

    shap_values = np.asarray(explanation.values)

    # SHAP summaries
    site_mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
        name="mean_abs_shap",
    ).sort_values(ascending=False)

    site_mean_shap = pd.Series(
        shap_values.mean(axis=0),
        index=feature_names,
        name="mean_shap",
    )

    cluster_mean_abs_shap = {}
    cluster_mean_shap = {}

    for cl in sorted(labels.unique(), key=str):
        idx = np.where(labels.to_numpy() == cl)[0]

        cluster_mean_abs_shap[cl] = pd.Series(
            np.abs(shap_values[idx]).mean(axis=0),
            index=feature_names,
            name=f"cluster_{cl}_mean_abs_shap",
        ).sort_values(ascending=False)

        cluster_mean_shap[cl] = pd.Series(
            shap_values[idx].mean(axis=0),
            index=feature_names,
            name=f"cluster_{cl}_mean_shap",
        )

    # SHAP-IQ aggregate summaries
    site_iv_mean = aggregate_interaction_values(
        ivs,
        aggregation="mean",
    )

    cluster_iv_mean = {}

    for cl in sorted(labels.unique(), key=str):
        idx = np.where(labels.to_numpy() == cl)[0]
        cluster_iv_mean[cl] = aggregate_interaction_values(
            [ivs[i] for i in idx],
            aggregation="mean",
        )

    plot_artifacts = {
        "target": target,
        "feature_comb": feature_comb,
        "site_name": "EMC",

        "feature_cols": feature_names,
        "label_replacements": label_replacements or LABEL_REPLACEMENTS,

        "labels": labels.to_numpy(),
        "cluster_counts": {
            str(k): int(v)
            for k, v in labels.value_counts(dropna=False).to_dict().items()
        },

        "cluster_rules": cluster_rules,
        "rule_info": rule_info,
        "ship_cluster_counts": ship_cluster_counts,

        "site_mean_abs_shap": site_mean_abs_shap,
        "site_mean_shap": site_mean_shap,
        "cluster_mean_abs_shap": cluster_mean_abs_shap,
        "cluster_mean_shap": cluster_mean_shap,

        "site_mean_interaction_values": site_iv_mean,
        "cluster_mean_interaction_values": cluster_iv_mean,

        "n_samples": int(len(labels)),

        "notes": (
            "This artifact contains EMC subgroup labels and derived SHAP/SHAP-IQ "
            "summaries. It does not include EMC raw/preprocessed feature values."
        ),
    }

    if include_subject_level_values:
        plot_artifacts["shap_values"] = shap_values.astype("float32")
        plot_artifacts["shapiq_ivs"] = ivs
        plot_artifacts["notes"] += (
            " Subject-level SHAP values and SHAP-IQ interaction values are included, "
            "but EMC raw/preprocessed feature values are still excluded."
        )

    artifact_path = os.path.join(
        out_dir,
        f"emc_subgroup_plot_artifacts_{target}_{feature_comb}.pkl",
    )
    with open(artifact_path, "wb") as f:
        pickle.dump(plot_artifacts, f)

    labels_path = os.path.join(
        out_dir,
        f"emc_subgroup_labels_{target}_{feature_comb}.csv",
    )
    pd.DataFrame(
        {
            "row_index": np.arange(len(labels)),
            "cluster": labels.to_numpy(),
        }
    ).to_csv(labels_path, index=False)

    summary_path = os.path.join(
        out_dir,
        f"emc_subgroup_summary_{target}_{feature_comb}.txt",
    )
    with open(summary_path, "w") as f:
        f.write(f"Target: {target}\n")
        f.write(f"Feature combination: {feature_comb}\n")
        f.write("Site: EMC\n")
        f.write(f"N samples: {len(labels)}\n\n")

        f.write("EMC subgroup counts:\n")
        for cl, n in labels.value_counts(dropna=False).items():
            f.write(f"  Cluster {cl}: n={n}\n")

        f.write("\nSHIP-derived cluster counts:\n")
        for cl, n in ship_cluster_counts.items():
            f.write(f"  Cluster {cl}: n={n}\n")

        f.write("\nRules applied:\n")
        for cl, rules in cluster_rules.items():
            f.write(f"\nCluster {cl}\n")
            for r in rules:
                f.write(f"  {r}\n")

        f.write("\nTop overall EMC mean |SHAP| features:\n")
        for feat, val in site_mean_abs_shap.head(20).items():
            f.write(f"  {feat}: {val:.6f}\n")

    print(f"Saved plot-ready artifact: {artifact_path}")
    print(f"Saved labels CSV: {labels_path}")
    print(f"Saved summary TXT: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply SHIP-derived subgroup rules to EMC for Sleep_Cov only."
    )

    parser.add_argument(
        "target",
        nargs="?",
        default="Stroop",
        choices=list(TARGET_TO_COLUMN.keys()),
        help="Prediction target. Default: Stroop",
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help=(
            "Path to EMC CSV. If not provided, the script tries "
            "Data/EMC_dataset_renamed_target_cleaned.csv, then Data/EMC.csv."
        ),
    )

    parser.add_argument(
        "--model_root",
        type=str,
        default=None,
        help=(
            "Root of AutoGluon model folder. Defaults to "
            "project_root/Code/Out-of-sample_validation/models/AutoGluon."
        ),
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=(
            "Output root. Defaults to "
            "project_root/Code/Out-of-sample_validation/Results/EMC_subgroup_analysis."
        ),
    )

    parser.add_argument("--max_display", type=int, default=20)
    parser.add_argument("--show", action="store_true")

    parser.add_argument(
        "--include_subject_level_values",
        action="store_true",
        help=(
            "Include subject-level SHAP values and SHAP-IQ objects in the saved "
            "plot artifact. Raw/preprocessed EMC feature values are still excluded."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    target = args.target
    feature_comb = FEATURE_COMB
    target_col = TARGET_TO_COLUMN[target]

    print(
        f"\nStarting EMC subgroup analysis → "
        f"target={target}, feature_comb={feature_comb}\n"
    )

    # --------------------------
    # Paths
    # --------------------------
    project_root = args.project_root
    code_root = os.path.join(project_root, "Code")
    oocv_root = os.path.join(code_root, "Out-of-sample_validation")
    data_root = os.path.join(project_root, "Data")

    sys.path.append(os.path.join(code_root, "lib"))
    sys.path.append(os.path.join(oocv_root, "lib"))

    import utils  # noqa

    model_root = args.model_root or os.path.join(
        oocv_root,
        "models",
        "AutoGluon",
    )

    results_root = args.results_root or os.path.join(
        oocv_root,
        "Results",
        "EMC_subgroup_analysis",
    )

    model_case_path = os.path.join(model_root, target, feature_comb)
    shap_path = os.path.join(model_case_path, "SHAP")
    shapiq_path = os.path.join(model_case_path, "shapiq")

    case_results_path = os.path.join(results_root, target, feature_comb)
    site_plot_dir = os.path.join(case_results_path, "site_plots")
    cluster_plot_dir = os.path.join(case_results_path, "cluster_plots")
    artifact_out_dir = os.path.join(case_results_path, "plot_artifacts")

    os.makedirs(case_results_path, exist_ok=True)

    print("Model case path:", model_case_path)
    print("Results path:", case_results_path)

    # --------------------------
    # Load subgroup artifact
    # --------------------------
    subgroup_artifact_path = os.path.join(
        model_case_path,
        f"subgroup_artifacts_{target}_{feature_comb}.pkl",
    )

    if not os.path.exists(subgroup_artifact_path):
        raise FileNotFoundError(
            f"Subgroup artifact not found: {subgroup_artifact_path}"
        )

    with open(subgroup_artifact_path, "rb") as f:
        subgroup_artifacts = pickle.load(f)

    feature_cols = list(subgroup_artifacts["feature_cols"])
    cluster_rules = subgroup_artifacts["cluster_rules"]
    rule_info = subgroup_artifacts.get("rule_info", {})
    ship_cluster_counts = subgroup_artifacts.get("ship_cluster_counts", {})
    label_replacements = subgroup_artifacts.get(
        "label_replacements",
        LABEL_REPLACEMENTS,
    )

    print("\nLoaded subgroup artifact.")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Clusters: {list(cluster_rules.keys())}")

    # --------------------------
    # Load EMC data
    # --------------------------
    if args.data_file is not None:
        data_file = args.data_file
    else:
        candidate_1 = os.path.join(data_root, "EMC_dataset_renamed_target_cleaned.csv")
        candidate_2 = os.path.join(data_root, "EMC.csv")

        if os.path.exists(candidate_1):
            data_file = candidate_1
        elif os.path.exists(candidate_2):
            data_file = candidate_2
        else:
            raise FileNotFoundError(
                "Could not find EMC data file. Tried:\n"
                f"  {candidate_1}\n"
                f"  {candidate_2}\n"
                "Use --data_file to specify the path."
            )

    print("\nLoading EMC data:", data_file)

    df_emc = pd.read_csv(data_file)

    if target_col not in df_emc.columns:
        raise KeyError(f"Target column '{target_col}' not found in EMC data.")

    df_emc = df_emc.dropna(subset=[target_col]).reset_index(drop=True)

    # --------------------------
    # Minimal Sleep_Cov preprocessing
    # --------------------------
    df_emc_ml = preprocess_emc_sleep_cov(
        df_emc,
        utils_module=utils,
        feature_cols=feature_cols,
    )

    validate_required_columns(
        df_emc_ml,
        feature_cols,
        name="EMC preprocessed dataframe",
    )

    # --------------------------
    # Apply subgroup rules
    # --------------------------
    labels_emc = apply_subgroup_rules(
        df_emc_ml,
        rules=cluster_rules,
        cols=feature_cols,
        unknown="unassigned",
    )

    # --------------------------
    # Load EMC SHAP / SHAP-IQ outputs
    # --------------------------
    shap_file = os.path.join(shap_path, "explanation_train_set_test_EMC.pkl")
    shapiq_file = os.path.join(shapiq_path, "ivs_SHIP_EMC.pkl")

    if not os.path.exists(shap_file):
        raise FileNotFoundError(f"EMC SHAP file not found: {shap_file}")

    if not os.path.exists(shapiq_file):
        raise FileNotFoundError(f"EMC SHAP-IQ file not found: {shapiq_file}")

    print("\nLoading EMC SHAP:", shap_file)
    with open(shap_file, "rb") as f:
        explanation_emc = pickle.load(f)

    print("Loading EMC SHAP-IQ:", shapiq_file)
    with open(shapiq_file, "rb") as f:
        ivs_emc = pickle.load(f)

    validate_row_alignment(
        df_site=df_emc_ml,
        labels=labels_emc,
        explanation=explanation_emc,
        ivs=ivs_emc,
    )

    validate_feature_alignment(
        explanation=explanation_emc,
        feature_cols=feature_cols,
    )

    # --------------------------
    # Plots
    # --------------------------
    make_emc_site_plots(
        explanation=explanation_emc,
        ivs=ivs_emc,
        df_emc_ml=df_emc_ml,
        feature_names=feature_cols,
        save_dir=site_plot_dir,
        label_replacements=label_replacements,
        max_display=args.max_display,
        show=args.show,
    )

    make_emc_cluster_plots(
        labels=labels_emc.values,
        explanation=explanation_emc,
        ivs=ivs_emc,
        feature_names=feature_cols,
        save_dir=cluster_plot_dir,
        label_replacements=label_replacements,
        max_display=args.max_display,
        show=args.show,
    )

    # --------------------------
    # Save derived outputs for Harvey
    # --------------------------
    save_plot_ready_artifacts(
        out_dir=artifact_out_dir,
        target=target,
        feature_comb=feature_comb,
        labels=labels_emc,
        explanation=explanation_emc,
        ivs=ivs_emc,
        feature_names=feature_cols,
        cluster_rules=cluster_rules,
        rule_info=rule_info,
        ship_cluster_counts=ship_cluster_counts,
        label_replacements=label_replacements,
        include_subject_level_values=args.include_subject_level_values,
    )

    print("\nDone. EMC subgroup analysis completed successfully.")
    print("Figures and artifacts saved under:", case_results_path)


if __name__ == "__main__":
    main()