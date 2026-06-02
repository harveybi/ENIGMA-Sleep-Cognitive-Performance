#!/bin/bash

# Output file for HTCondor submit
submit_file="recon-all_Juelich_FreeSurfer.submit"

# Clear the existing submit file if it exists
> $submit_file

# Define the number of cores (CPUs)
num_cores=1

# Write the environment settings to the submit file
echo "# The environment" >> $submit_file
echo "universe       = vanilla" >> $submit_file
echo "getenv         = True" >> $submit_file
echo "request_memory = 4GB" >> $submit_file
echo "request_disk   = 2GB" >> $submit_file
echo "request_cpus   = $num_cores" >> $submit_file  # Here, use the num_cores variable
echo "" >> $submit_file

# Write the execution settings to the submit file
echo "# Execution" >> $submit_file
echo "executable = /data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/recon-all_Juelich_FreeSurfer.sh" >> $submit_file
echo "" >> $submit_file

# Define common variables
log_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/condor_logs/Juelich_FreeSurfer"
output_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/condor_outputs/Juelich_FreeSurfer"
error_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/condor_errors/Juelich_FreeSurfer"

# get subject list from /data/project/sleep_ENIGMA_insomnia/Data/Chongqing_BIDS. Subjects are in the format of sub-xxx, it's the name of the folder
#subjects=($(ls /data/project/sleep_ENIGMA_insomnia/Data/Chongqing_BIDS))
# another way, subject is just the name of the file. like the xxx of xxx.nii
subjects=($(ls /data/project/sleep_ENIGMA_harmonization/data/Juelich_FreeSurfer/*.nii.gz | xargs -n 1 basename | cut -d '_' -f 1))

# Loop through subjects
for subject in ${subjects[@]}; do
      echo "arguments = ${subject}" >> $submit_file  # Here, use the num_cores variable
      echo "log       = ${log_dir}/${subject}.log" >> $submit_file
      echo "output    = ${output_dir}/${subject}.out" >> $submit_file
      echo "error     = ${error_dir}/${subject}.err" >> $submit_file
      echo "Queue" >> $submit_file
      echo "" >> $submit_file
done
