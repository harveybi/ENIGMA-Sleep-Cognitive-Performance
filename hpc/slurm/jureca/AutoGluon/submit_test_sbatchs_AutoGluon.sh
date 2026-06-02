#!/bin/bash

arg1="Sleep_Cov"
arg2="Stroop"

# Export the variables so they can be used in the SLURM script
export ARG1=$arg1
export ARG2=$arg2

# Call sbatch
sbatch /p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/test_sbatch_AutoGluon_SHIP_Stroop_NAI.sh