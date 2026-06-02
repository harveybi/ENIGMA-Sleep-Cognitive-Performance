# Analysis Workflow

1. Configure local paths in `configs/paths.local.yml`.
2. Run cohort-specific preparation scripts in `scripts/01_data_preparation/`.
3. Train models from `scripts/02_model_training/`.
4. Run external validation scripts from `scripts/03_out_of_sample_validation/`.
5. Run SHAP and SHAP-IQ scripts from `scripts/04_shap_analysis/`.
6. Regenerate manuscript figures and tables with `scripts/05_figures_tables/`.
7. Use `hpc/` templates for large-scale cluster execution.

The repository is organized so that scripts are reviewable without including protected data or generated model artifacts.

