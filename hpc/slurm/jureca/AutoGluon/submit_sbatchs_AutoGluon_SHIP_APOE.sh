#!/bin/bash

# Define the feature combinations and targets
feature_combs=("Sleep_APOE" "Sleep_APOE_Shuffle"
               "Sleep_Cov_APOE" "Sleep_Cov_APOE_Shuffle"
               "Cov_APOE" "Cov_APOE_Shuffle"
               "Brain_APOE" "Brain_APOE_Shuffle"
               "Subcor_APOE" "Subcor_APOE_Shuffle"
               "Sleep_Cov_Brain_APOE" "Sleep_Cov_Brain_APOE_Shuffle"
               "Sleep_Cov_Subcor_APOE" "Sleep_Cov_Subcor_APOE_Shuffle")
targets=("Stroop" "Memory")

# Iterate through each combination
for arg1 in "${feature_combs[@]}"; do
  for arg2 in "${targets[@]}"; do
    # Export the variables so they can be used in the SLURM script
    export ARG1=$arg1
    export ARG2=$arg2

    # Call sbatch
    sbatch /p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/sbatch_AutoGluon_SHIP_APOE_Stroop_NAI.sh

  done
done