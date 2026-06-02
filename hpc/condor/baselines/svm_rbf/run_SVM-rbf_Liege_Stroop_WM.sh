#!/bin/bash
source /home/h.bi/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate new_autogluon

# Set the environment variables
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python3 /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/svm_rbf/SVM-rbf_Liege_Stroop_WM.py "$@" >> "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/svm_rbf/combined_logs/Liege/${2}_${1}_log.txt" 2>&1