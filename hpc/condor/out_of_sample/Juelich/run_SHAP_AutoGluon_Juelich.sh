#!/bin/bash
#source /home/h.bi/miniforge3/etc/profile.d/conda.sh

#mamba deactivate          # harmless if nothing is active
mamba activate new_autogluon

# Set the environment variables
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python3 /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Juelich/SHAP_AutoGluon_Juelich.py "$@" >> "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/combined_logs/Juelich/SHAP/${2}_${1}_${3}_log.txt" 2>&1