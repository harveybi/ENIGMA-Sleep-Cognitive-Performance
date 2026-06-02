#!/bin/bash
source /home/h.bi/miniforge3/etc/profile.d/conda.sh

conda deactivate
#conda activate new_autogluon
conda activate gpu_autogluon

# Set the environment variables
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python3 /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Liege/SHAP_AutoGluon_Liege_no_DK.py "$@" >> "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/combined_logs/Liege/SHAP_no_DK/${2}_${1}_log.txt" 2>&1