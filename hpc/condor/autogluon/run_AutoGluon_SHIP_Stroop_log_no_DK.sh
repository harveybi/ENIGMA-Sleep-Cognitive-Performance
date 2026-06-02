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

LOG="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/combined_logs/SHIP/ml_pipeline_no_DK_log/${2}_${1}_log.txt"

python3 /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/AutoGluon_SHIP_Stroop_log_no_DK.py "$@" >> "$LOG" 2>&1
rc=$?
ts=$(date '+%F %T')

if [ $rc -eq 0 ]; then
  echo "[$ts] RUN FINISHED: ${2} × ${1} (exit=0)"
else
  echo "[$ts] RUN FAILED:   ${2} × ${1} (exit=$rc) — see log: $LOG"
fi

exit $rc