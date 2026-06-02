# Reproducibility Checklist

This checklist maps the repository to common Nature Portfolio machine-learning reporting expectations.

## Code

- Source code is under `src/enigma_sleep_cognition/`.
- Analysis entrypoints are under `scripts/`.
- HPC submission templates are under `hpc/`.
- Reviewer smoke test: `bash scripts/run_smoke_test.sh`.

## Data

- Raw human participant-level data are not committed.
- Controlled-access data requirements are documented in `DATA_AVAILABILITY.md`.
- Synthetic test data are available under `data/example/`.
- Local data paths should be configured through `configs/paths.local.yml`.

## Models

- Trained models are not tracked in Git because they are large generated artifacts.
- Model-generation scripts and submission templates are included.
- Final model archives should be deposited externally and cited by DOI when available.

## Environment

- Full conda environment: `environment.yml`.
- Minimal smoke-test dependencies: `requirements.txt`.
- Python package metadata: `pyproject.toml`.

## Results

- Compact summary files are kept under `results/summaries/`.
- Selected manuscript-level figures/tables are kept under `results/manuscript/`.
- Bulk intermediate outputs are excluded and should be regenerated or archived externally.

## Limitations

- The synthetic smoke test checks software wiring only; it does not reproduce manuscript-level performance.
- Full reproduction requires controlled-access cohort data and HPC resources.

