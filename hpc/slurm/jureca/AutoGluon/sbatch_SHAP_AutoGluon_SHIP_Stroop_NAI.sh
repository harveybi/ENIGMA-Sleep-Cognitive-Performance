#!/bin/bash -x
#SBATCH --account=inm7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=dc-cpu
#SBATCH --output=logs/outputs/%x_%j.out
#SBATCH --error=logs/errors/%x_%j.err

source /p/project/cinm-7/bi1/miniconda3/etc/profile.d/conda.sh

conda deactivate
conda activate new_autogluon

# Set the environment variables
#export MKL_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
#export OMP_NUM_THREADS=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} python3 /p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/SHAP_AutoGluon_SHIP_Stroop_NAI.py "$ARG1" "$ARG2" >> "/p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/combined_logs/SHAP/${ARG2}_${ARG1}_log.txt" 2>&1