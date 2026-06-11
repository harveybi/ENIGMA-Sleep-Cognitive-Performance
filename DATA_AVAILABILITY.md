# Data Availability

Raw participant-level data are not included in this repository. The analysis uses controlled-access human cohort data and site-specific neuroimaging/cognitive variables that may require data-use agreements, ethics approvals, or cohort-specific permission.

Datasets/cohorts referenced by the analysis scripts include:

- SHIP / Greifswald
- EMC / Rotterdam
- KI
- Liege
- Juelich
- Pitts
- VETSA

No demo dataset is provided because manuscript-scale analyses require controlled-access human cohort data.

Expected local paths are documented in `configs/paths.example.yml`. Users with approved access should copy that file to `configs/paths.local.yml` and point each entry to local controlled-access data locations.

Atlas and external reference files:

- Large neuroimaging atlas volumes, including Schaefer NIfTI files, are not tracked and should be downloaded from their original atlas source or provided through the project data archive.
- External reference files used for map or brain visualizations are not raw participant data and should be retrieved from their original sources or the project archive when needed.
