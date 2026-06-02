#!/bin/bash

# Output file for HTCondor submit
submit_file="submit_run_SHAP-IQ_AutoGluon_Liege.submit"

# Clear the existing submit file if it exists
> $submit_file

# Define the number of cores (CPUs)
num_cores=1

# Write the environment settings to the submit file
echo "# The environment" >> $submit_file
echo "universe       = vanilla" >> $submit_file
echo "getenv         = True" >> $submit_file
echo "request_memory = 2GB" >> $submit_file
echo "request_disk   = 1GB" >> $submit_file
echo "request_cpus   = $num_cores" >> $submit_file  # Here, use the num_cores variable
echo "" >> $submit_file

# Write the execution settings to the submit file
echo "# Execution" >> $submit_file
echo "executable = /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Liege/run_SHAP-IQ_AutoGluon_Liege.sh" >> $submit_file
echo "" >> $submit_file

# Define common variables
log_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/condor_errors/SHAP-IQ/Liege"
output_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/condor_outputs/SHAP-IQ/Liege"
error_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/condor_errors/SHAP-IQ/Liege"

# Define the list of feature combinations, zscore cases, and targets
feature_combs=("Sleep" "Cov" "Sleep_Cov" "Sleep_Shuffle_Cov" "Brain" "Sleep_Cov_Brain" "Sleep_Cov_Brain_Shuffle" "Sleep_APOE" "Sleep_APOE_Shuffle" "Sleep_Cov_APOE" "Sleep_Cov_APOE_Shuffle" "Sleep_Cov_Brain_APOE" "Sleep_Cov_Brain_APOE_Shuffle")
targets=("Stroop" "Memory" "Stroop_rgo_age" "Memory_rgo_age")

# Loop through each combination of feature_comb, zscore_case, and target
for target in "${targets[@]}"; do
  for feature_comb in "${feature_combs[@]}"; do
    # Apply condition: if the target contains '_rgo_age', limit the features
    if [[ "$target" == *_rgo_age ]] && [[ "$feature_comb" != "Sleep_Cov" && "$feature_comb" != "Sleep_Cov_Brain" ]]; then
      continue # Skip this iteration
    fi

    echo "arguments = ${feature_comb} ${target}" >> $submit_file  # Here, use the num_cores variable
    echo "log       = ${log_dir}/${target}_${feature_comb}.log" >> $submit_file
    echo "output    = ${output_dir}/${target}_${feature_comb}.out" >> $submit_file
    echo "error     = ${error_dir}/${target}_${feature_comb}.err" >> $submit_file
    echo "Queue" >> $submit_file
    echo "" >> $submit_file

  done
done
