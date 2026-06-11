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
- Typical installation time is approximately 20-40 minutes for the main analysis environment, depending on conda solver speed and network connection.

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

## Full Analysis Workflow

1. Configure local data paths by copying `configs/paths.example.yml` to `configs/paths.local.yml`.
2. Prepare cohort-specific tabular and imaging-derived features with `scripts/01_data_preparation/`.
3. Train primary models with `scripts/02_model_training/autogluon/` and comparator models with `xgboost/` and `baselines/`.
4. Run external validation scripts in `scripts/03_out_of_sample_validation/`.
5. Generate SHAP and SHAP-IQ outputs with `scripts/04_shap_analysis/`.
6. Recreate manuscript figures and tables with `scripts/05_figures_tables/`.
7. Use `hpc/` submission templates for large-scale HTCondor or SLURM execution.

## Data Requirements

Raw participant-level data, site-level clinical variables, and protected cohort files are not committed. See `DATA_AVAILABILITY.md` and `configs/paths.example.yml` for controlled-access details and expected local directory layout.

No demo dataset is provided because manuscript-scale analyses require controlled-access human cohort data. For the Nature code checklist, the demo dataset, demo expected output, and demo runtime items should therefore be left unchecked.

## Citation

TBA

## Contact me

Github issue or h.bi@fz-juelich.de
