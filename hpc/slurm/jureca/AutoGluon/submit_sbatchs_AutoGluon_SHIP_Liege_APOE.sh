#!/bin/bash

# Define the feature combinations and targets
feature_combs=("Sleep" "Cov" "Brain" "CT" "SA" "Subcor" "Sleep_Cov"
               "Sleep_Cov_Brain" "Sleep_Cov_CT" "Sleep_Cov_SA" "Sleep_Cov_Subcor"
               "Sleep_Brain" "Sleep_CT" "Sleep_SA" "Sleep_Subcor"
               "Cov_Brain" "Cov_CT" "Cov_SA" "Cov_Subcor"
               "Sleep_Shuffle_Cov" "Sleep_Cov_Subcor_Shuffle" "Sleep_Shuffle_Cov_Subcor"
               "Sleep_Shuffle_Subcor" "Cov_Brain_Shuffle" "Cov_Subcor_Shuffle"
               "Sleep_APOE" "Sleep_APOE_Shuffle"
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
    sbatch /p/project/cinm-7/bi1/ENIGMA_Sleep_Cog_Prediction/Code/AutoGluon/sbatch_AutoGluon_SHIP_Liege_APOE_Stroop_NAI.sh

  done
done