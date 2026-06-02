#!/bin/bash -x
#SBATCH --account=training2410
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --partition=dc-cpu
#SBATCH --job-name=$ARG1_$ARG2

# Handle dynamic log paths inside the script
out_log="logs/outputs/${ARG2}_${ARG1}.out"
err_log="logs/errors/${ARG2}_${ARG1}.err"

# Redirect SLURM output and error logs
exec 1> $out_log
exec 2> $err_log

source /p/project/cinm-7/bi1/miniconda3/etc/profile.d/conda.sh

conda deactivate
conda activate new_autogluon

# Set the environment variables
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

#srun python3 /p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/AutoGluon_SHIP_Stroop_NAI.py "$ARG1" "$ARG2" "${SLURM_CPUS_PER_TASK}" >> "/p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/combined_logs/ml_pipeline/${ARG2}_${ARG1}_log.txt" 2>&1

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} python3 /p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/AutoGluon_SHIP_Stroop_NAI.py "$ARG1" "$ARG2" "${SLURM_CPUS_PER_TASK}" >> "/p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/combined_logs/ml_pipeline/${ARG2}_${ARG1}_log.txt" 2>&1