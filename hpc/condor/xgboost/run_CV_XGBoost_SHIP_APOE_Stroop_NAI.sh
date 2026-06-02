#!/bin/bash
source /home/h.bi/anaconda3/etc/profile.d/conda.sh

conda deactivate
conda activate new_autogluon

# Set the environment variables
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Set the directory path
combine_log_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/XGBoost/combined_logs/SHIP/cv/${2}_${1}/"

# Check if the directory exists and create it if not
mkdir -p "$combine_log_dir"

python3 /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/XGBoost/CV_XGBoost_SHIP_APOE_Stroop_NAI.py "${1}" "${2}" "${3}" "${4}" "${5}" >> "${combine_log_dir}${3}_${4}_log.txt" 2>&1
