# ENIGMA Sleep Cognition

This repository contains the analysis code for the manuscript “Prediction of cognitive performance by demographics, sleep, and brain morphometry: machine learning findings from the ENIGMA-Sleep Working Group.” The code supports multisite cohort data preparation, leakage-controlled machine learning model training, out-of-cohort validation, SHAP and SHAP-IQ model interpretation, participant subgroup analysis.

## Repository Map

```text
src/enigma_sleep_cognition/                 reusable Python pipeline modules
scripts/01_data_preparation/                cohort preparation and neuroimaging extraction scripts
scripts/02_model_training/                  AutoGluon, XGBoost, and baseline model scripts
scripts/03_out_of_sample_validation/        external-site validation scripts
scripts/04_shap_analysis/                   SHAP and SHAP-IQ interpretation scripts
scripts/05_figures_tables/                  manuscript figure/table generation scripts
hpc/condor/                                 HTCondor submission templates and DAGMan files
hpc/slurm/                                  JURECA/SLURM submission templates
configs/                                    example path and model-set configuration files
docs/                                       reproducibility and workflow notes
notebooks/                                  optional notebook location; currently documentation only
```

## System Requirements

- Operating system: Linux x86_64 tested on HPC/workstation environments.
- Main analysis environment: Python 3.10.18.
- Brain visualization environment: Python 3.11.14.
- No non-standard hardware is required to inspect or install the code.
- Full manuscript-scale reproduction requires controlled-access cohort data and HPC resources for large-scale model training, SHAP, SHAP-IQ, and out-of-cohort validation.

## Installation

Create the main analysis environment:

```bash
conda env create -f environment-analysis.yml
conda activate enigma-sleep-cognition-analysis
python -m pip install -e . --no-deps
```

Create the optional brain visualization environment:

```bash
conda env create -f environment-brainviz.yml
conda activate enigma-sleep-cognition-brainviz
```

## Using the Scripts With Local Data

This repository does not include demo data. The scripts are manuscript analysis entry points that can be run after controlled-access cohort data have been obtained and placed in the expected local directory structure.

1. Use `configs/paths.example.yml` as a template for the local data, neuroimaging, output, and log paths needed for execution.
2. Adapt script-local path variables such as `data_save_path`, `raw_data_save_path`, `results_path`, and `trained_models_path` to point to the local controlled-access data and output directories.
3. Run cohort preparation scripts in `scripts/01_data_preparation/` to generate harmonized analysis tables.
4. Run model-training scripts in `scripts/02_model_training/` after the prepared cohort tables are available.
5. Run external validation scripts in `scripts/03_out_of_sample_validation/` when validation cohort data are available.
6. Run SHAP and SHAP-IQ scripts in `scripts/04_shap_analysis/` after trained model artifacts have been generated.
7. Run figure and table scripts in `scripts/05_figures_tables/` after model, validation, and interpretation outputs are available.

For scripts with command-line arguments, pass the requested feature set, target, and compute settings after local paths have been adapted. For example:

```bash
python scripts/02_model_training/xgboost/XGBoost_SHIP_Stroop_NAI.py Sleep_Cov Stroop 8
```

## Full Analysis Workflow

1. Configure local data, neuroimaging, output, and log paths using `configs/paths.example.yml` as a template.
2. Prepare cohort-specific tabular and imaging-derived features with `scripts/01_data_preparation/`.
3. Train primary models with `scripts/02_model_training/autogluon/` and comparator models with `xgboost/` and `baselines/`.
4. Run external validation scripts in `scripts/03_out_of_sample_validation/`.
5. Generate SHAP and SHAP-IQ outputs with `scripts/04_shap_analysis/`.
6. Recreate manuscript figures and tables with `scripts/05_figures_tables/`.
7. Use `hpc/` submission templates for large-scale HTCondor or SLURM execution.

## Data Requirements

Raw participant-level data, site-level clinical variables, and protected cohort files are not committed. See `DATA_AVAILABILITY.md` and `configs/paths.example.yml` for controlled-access details and expected local directory layout.

No demo dataset is provided because manuscript-scale analyses require controlled-access cohort data. The workflow instructions above describe how to run the scripts with authorized local data. They should not be interpreted as demo data, demo expected output, or demo runtime.

## Citation

TBA

## Contact me

Github issue or h.bi@fz-juelich.de
