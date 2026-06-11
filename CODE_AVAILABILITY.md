# Code Availability

All analysis scripts for this study are provided in this repository:

https://github.com/harveybi/ENIGMA-Sleep-Cognitive-Performance

The study was preregistered on the Open Science Framework:

https://osf.io/nhbkq/

## License

The source code is distributed under the MIT License. See `LICENSE`.

## Environment Specifications

Two Python environments are provided:

- `environment-analysis.yml`: main statistical, machine-learning, SHAP, SHAP-IQ, subgroup, and out-of-cohort validation analyses.
- `environment-brainviz.yml`: brain visualization workflows.

The environment versions correspond to the software versions reported in the manuscript. The `requirements.txt` file provides a pip-formatted dependency list for the main analysis environment.

## Data and Models

Raw human participant-level data are not included in this repository. Trained models and large intermediate outputs are also not tracked because they are generated artifacts and may depend on controlled-access cohort data.

No demo dataset is provided because manuscript-scale analyses require controlled-access human cohort data. The repository instead provides source code, environment specifications, configuration templates, and workflow documentation.

## Archival Release

The GitHub repository is the active code location. A versioned archival release and DOI can be created from the final manuscript-associated GitHub release if required by the journal or upon acceptance.
