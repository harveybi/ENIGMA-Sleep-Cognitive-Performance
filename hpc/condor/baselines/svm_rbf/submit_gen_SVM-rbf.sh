#!/bin/bash

# Output file for HTCondor submit
submit_file="submit_SVM-rbf_SHIP_Stroop_NAI.submit"

# Clear the existing submit file if it exists
> $submit_file

# Define the number of cores (CPUs)
num_cores=4

# Write the environment settings to the submit file
echo "# The environment" >> $submit_file
echo "universe       = vanilla" >> $submit_file
echo "getenv         = True" >> $submit_file
echo "request_memory = 64GB" >> $submit_file
echo "request_disk   = 10GB" >> $submit_file
echo "request_cpus   = $num_cores" >> $submit_file  # Here, use the num_cores variable
echo "" >> $submit_file

# Write the execution settings to the submit file
echo "# Execution" >> $submit_file
echo "executable = /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/svm_rbf/run_SVM-rbf_SHIP_Stroop_NAI.sh" >> $submit_file
echo "" >> $submit_file

# Define common variables
log_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/svm_rbf/condor_logs/SHIP"
output_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/svm_rbf/condor_outputs/SHIP"
error_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/svm_rbf/condor_errors/SHIP"

# Define the list of feature combinations, zscore cases, and targets
feature_combs=("Sleep" "Cov" "Sleep_Cov" "Sleep_Cov_Brain" "Sleep_Cov_CT" "Sleep_Cov_SA" "Sleep_Cov_Subcor" "Cov_Brain" "Cov_CT" "Cov_SA" "Cov_Subcor" "Sleep_Shuffle_Cov" "Sleep_Cov_Subcor_Shuffle" "Cov_Brain_Shuffle" "Cov_Subcor_Shuffle")
targets=("Stroop" "Memory")

# Loop through each combination of feature_comb, zscore_case, and target
for target in "${targets[@]}"; do
    for feature_comb in "${feature_combs[@]}"; do
        echo "arguments = ${feature_comb} ${target} $num_cores" >> $submit_file  # Here, use the num_cores variable
        echo "log       = ${log_dir}/${target}_${feature_comb}.log" >> $submit_file
        echo "output    = ${output_dir}/${target}_${feature_comb}.out" >> $submit_file
        echo "error     = ${error_dir}/${target}_${feature_comb}.err" >> $submit_file
        echo "Queue" >> $submit_file
        echo "" >> $submit_file
    done
done
