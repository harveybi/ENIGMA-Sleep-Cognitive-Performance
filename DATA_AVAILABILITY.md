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

The repository provides synthetic example data in `data/example/` only for software smoke testing. These files are not intended to reproduce manuscript results.

Expected local paths are documented in `configs/paths.example.yml`. Users with approved access should copy that file to `configs/paths.local.yml` and point each entry to local controlled-access data locations.

Atlas and external reference files:

- Natural Earth map data used for figures are included under `data/external/natural_earth/`.
- Large neuroimaging atlas volumes, including Schaefer NIfTI files, are not tracked and should be downloaded from their original atlas source or provided through the project data archive.

