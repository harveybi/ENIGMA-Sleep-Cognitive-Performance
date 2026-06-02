#!/bin/bash
source /p/project/cinm-7/bi1/miniconda3/etc/profile.d/conda.sh

conda deactivate
conda activate new_autogluon

# Set the environment variables
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python3 /p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/AutoGluon_SHIP_Stroop_NAI.py "$@" >> "/p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/combined_logs/ml_pipeline/${2}_${1}_log.txt" 2>&1