# ENIGMA Sleep Cognition

This repository contains the analysis code, lightweight examples, and reviewer-facing reproducibility materials for the manuscript "Sleep, brain structure, and cognition across ENIGMA cohorts" (working title). The code supports multisite data preparation, machine-learning model training, out-of-sample validation, SHAP/SHAP-IQ interpretation, and manuscript figure/table generation. Participant-level cohort data are not included.

## Citation

Manuscript citation placeholder:

> Bi H, et al. Sleep, brain structure, and cognition across ENIGMA cohorts. Nature Mental Health. Forthcoming.

Software citation details are provided in `CITATION.cff`.

## Repository Map

```text
src/enigma_sleep_cognition/          reusable Python pipeline modules
scripts/01_data_preparation/         cohort preparation and neuroimaging extraction scripts
scripts/02_model_training/           AutoGluon, XGBoost, and baseline model scripts
scripts/03_out_of_sample_validation/ external-site validation scripts
scripts/04_shap_analysis/            SHAP and SHAP-IQ interpretation scripts
scripts/05_figures_tables/           manuscript figure/table generation scripts
hpc/                                 HTCondor and SLURM submission templates
configs/                             example path and model-set configuration
data/example/                        synthetic data for smoke testing
results/summaries/                   small summary tables retained for review
results/manuscript/                  selected manuscript-level outputs
docs/                                reproducibility and workflow notes
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate XGBoost
python -m pip install -e .
```

For a minimal smoke-test-only environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Smoke Test

Run the lightweight synthetic-data test from the repository root:

```bash
bash scripts/run_smoke_test.sh
```

Expected outputs:

```text
results/example_outputs/smoke_test_metrics.csv
results/example_outputs/smoke_test_prediction_plot.svg
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

Raw participant-level data, site-level clinical variables, and protected cohort files are not committed. See `DATA_AVAILABILITY.md` and `data/README.md` for controlled-access details and expected local directory layout.

## Results

This repository keeps small, interpretable summary files and selected manuscript outputs. Trained models, large SHAP HTML files, bulk plots, scheduler logs, and intermediate outputs are intentionally excluded and should be regenerated or deposited in an external archive such as Zenodo/OSF when finalized.
