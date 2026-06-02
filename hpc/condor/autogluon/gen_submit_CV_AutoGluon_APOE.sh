#!/bin/bash

# Output file for HTCondor submit
submit_file="submit_CV_AutoGluon_SHIP_APOE_Stroop_NAI.submit"

# Clear the existing submit file if it exists
> $submit_file

# Define the number of cores (CPUs)
num_cores=1

# Write the environment settings to the submit file
echo "# The environment" >> $submit_file
echo "universe       = vanilla" >> $submit_file
echo "getenv         = True" >> $submit_file
echo "request_memory = 32GB" >> $submit_file
echo "request_disk   = 32GB" >> $submit_file
echo "request_cpus   = $num_cores" >> $submit_file  # Here, use the num_cores variable
echo "" >> $submit_file

# Write the execution settings to the submit file
echo "# Execution" >> $submit_file
echo "executable = /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/run_CV_AutoGluon_SHIP_Stroop_NAI.sh" >> $submit_file
echo "" >> $submit_file

# Define common variables
log_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/condor_logs/SHIP/CV"
output_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/condor_outputs/SHIP/CV"
error_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/condor_errors/SHIP/CV"

# Define the list of feature combinations, zscore cases, and targets
feature_combs=("Sleep_APOE" "Sleep_APOE_Shuffle" "Sleep_Cov_APOE" "Sleep_Cov_APOE_Shuffle" "Cov_APOE" "Cov_APOE_Shuffle" "Brain_APOE" "Brain_APOE_Shuffle" "Subcor_APOE" "Subcor_APOE_Shuffle" "Sleep_Cov_Brain_APOE" "Sleep_Cov_Brain_APOE_Shuffle" "Sleep_Cov_Subcor_APOE" "Sleep_Cov_Subcor_APOE_Shuffle")
targets=("Stroop" "Memory")

# Loop through each combination of feature_comb, zscore_case, and target
for target in "${targets[@]}"; do
    for feature_comb in "${feature_combs[@]}"; do
        echo "arguments = ${feature_comb} ${target}" >> $submit_file  # Here, use the num_cores variable
        echo "log       = ${log_dir}/${target}_${feature_comb}.log" >> $submit_file
        echo "output    = ${output_dir}/${target}_${feature_comb}.out" >> $submit_file
        echo "error     = ${error_dir}/${target}_${feature_comb}.err" >> $submit_file
        echo "Queue" >> $submit_file
        echo "" >> $submit_file
    done
done
