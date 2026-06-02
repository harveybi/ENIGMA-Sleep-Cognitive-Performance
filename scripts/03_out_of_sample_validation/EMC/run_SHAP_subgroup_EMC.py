import os
import sys
import subprocess
from pathlib import Path


def run_script(
    script_path,
    feature,
    target,
    log_file,
    project_root=None,
    data_file=None,
    model_root=None,
    results_root=None,
    feature_lists=None,
    max_display=20,
    show=False,
    include_subject_level_values=False,
):
    cmd = [
        sys.executable,
        script_path,
        target,
        "--max_display", str(max_display),
    ]

    if project_root is not None:
        cmd += ["--project_root", project_root]

    if data_file is not None:
        cmd += ["--data_file", data_file]

    if model_root is not None:
        cmd += ["--model_root", model_root]

    if results_root is not None:
        cmd += ["--results_root", results_root]

    if feature_lists is not None:
        cmd += ["--feature_lists", feature_lists]

    if show:
        cmd += ["--show"]

    if include_subject_level_values:
        cmd += ["--include_subject_level_values"]

    with open(log_file, "a", buffering=1) as log:
        log.write(f"\n\n=== Running EMC subgroup analysis: feature={feature}, target={target} ===\n")
        log.write("Command: " + " ".join(cmd) + "\n\n")

        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"SHAP subgroup script failed for feature={feature}, target={target}. "
            f"Check log: {log_file}"
        )


def main():
    # ============================================================
    # EMC USER: change these paths to match your local/server setup
    # ============================================================

    # Full path to SHAP_subgroup_EMC.py
    script_path = (
        "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/"
        "Code/Out-of-sample_validation/EMC/SHAP_subgroup_EMC.py"
    )

    # Project root containing Data/ and Code/
    project_root = (
        "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive"
    )

    # Optional: explicitly set EMC data file.
    # If None, SHAP_subgroup_EMC.py will try:
    #   Data/EMC_dataset_renamed_target_cleaned.csv
    #   Data/EMC.csv
    data_file = None
    # Example:
    # data_file = "/path/to/EMC_dataset_renamed_target_cleaned.csv"

    # Optional: explicitly set AutoGluon model root.
    # If None, SHAP_subgroup_EMC.py will use:
    #   project_root/Code/Out-of-sample_validation/models/AutoGluon
    model_root = None
    # Example:
    # model_root = "/path/to/models/AutoGluon"

    # Optional: explicitly set output root.
    # If None, SHAP_subgroup_EMC.py will use:
    #   project_root/Code/Out-of-sample_validation/Results/EMC_subgroup_analysis
    results_root = None
    # Example:
    # results_root = "/path/to/Results/EMC_subgroup_analysis"

    # Optional: explicitly set feature_lists.pkl.
    # If None, SHAP_subgroup_EMC.py will use:
    #   project_root/Code/Out-of-sample_validation/feature_lists.pkl
    feature_lists = None
    # Example:
    # feature_lists = "/path/to/feature_lists.pkl"

    # Log directory
    log_dir = (
        "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/"
        "Code/Out-of-sample_validation/logs/subgroup_EMC"
    )

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Fixed analysis setting
    # ============================================================
    feature_combines = ["Sleep_Cov"]
    targets = ["Stroop"]

    # Plot options
    max_display = 20
    show = False

    # If True, output artifact will include subject-level SHAP values
    # and SHAP-IQ objects, but still not EMC raw/preprocessed feature values.
    include_subject_level_values = False

    for feature in feature_combines:
        for target in targets:
            log_file = os.path.join(log_dir, f"{target}_{feature}_subgroup_log.txt")

            print(f"Running EMC subgroup analysis: feature={feature}, target={target}")
            print(f"Log file: {log_file}")

            run_script(
                script_path=script_path,
                feature=feature,
                target=target,
                log_file=log_file,
                project_root=project_root,
                data_file=data_file,
                model_root=model_root,
                results_root=results_root,
                feature_lists=feature_lists,
                max_display=max_display,
                show=show,
                include_subject_level_values=include_subject_level_values,
            )


if __name__ == "__main__":
    main()