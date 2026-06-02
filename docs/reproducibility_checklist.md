# Reproducibility Checklist

This checklist maps the repository to common Nature Portfolio machine-learning reporting expectations.

## Code

- Source code is under `src/enigma_sleep_cognition/`.
- Analysis entrypoints are under `scripts/`.
- HPC submission templates are under `hpc/`.

## Data

- Raw human participant-level data are not committed.
- Controlled-access data requirements are documented in `DATA_AVAILABILITY.md`.
- Local data paths should be configured through `configs/paths.local.yml`.

## Models

- Trained models are not tracked in Git because they are large generated artifacts.
- Model-generation scripts and submission templates are included.
- Final model archives should be deposited externally and cited by DOI when available.

## Environment

- Full conda environment: `environment.yml`.
- Python dependencies are summarized in `requirements.txt`.
- Python package metadata: `pyproject.toml`.

## Results

- Compact summary files are kept under `results/summaries/`.
- Selected manuscript-level figures/tables are kept under `results/manuscript/`.
- Bulk intermediate outputs are excluded and should be regenerated or archived externally.

## Limitations

- Full reproduction requires controlled-access cohort data and HPC resources.
