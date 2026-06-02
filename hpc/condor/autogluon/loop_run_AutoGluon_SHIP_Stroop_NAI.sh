#!/bin/bash
source /home/h.bi/anaconda3/etc/profile.d/conda.sh

conda deactivate
conda activate new_autogluon

# Set the environment variables
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Define feature combinations and targets
feature_comb_list=("Sleep" "Cov" "Sleep_Cov" "Sleep_Shuffle_Cov" "Brain" "Sleep_Cov_Brain" "Sleep_Cov_Brain_Shuffle" "Sleep_APOE" "Sleep_APOE_Shuffle" "Sleep_Cov_APOE" "Sleep_Cov_APOE_Shuffle" "Sleep_Cov_Brain_APOE" "Sleep_Cov_Brain_APOE_Shuffle")
target_list=("Stroop" "Memory" "Stroop_rgo_age" "Memory_rgo_age")

# Directory for logs
log_dir="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/combined_logs/SHIP/ml_pipeline"
script_path="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/AutoGluon/AutoGluon_SHIP_Stroop_NAI.py"

# Loop through feature combinations and targets
for feature in "${feature_comb_list[@]}"; do
    for target in "${target_list[@]}"; do
        # Apply condition: if the target contains '_rgo_age', limit the features
        if [[ "$target" == *_rgo_age ]] && [[ "$feature" != "Sleep_Cov" && "$feature" != "Sleep_Cov_Brain" ]]; then
            continue # Skip this iteration
        fi

        # Define log file path
        log_file="${log_dir}/${target}_${feature}_log.txt"

        # Get the start time
        start_time=$(date +%s)
        echo "[$(date)] Starting script for feature: $feature, target: $target"

        # Execute the Python script and redirect output to log file
        python3 "$script_path" "$feature" "$target" 40 >> "$log_file" 2>&1

        # Get the end time
        end_time=$(date +%s)
        echo "[$(date)] Finished script for feature: $feature, target: $target"

        # Calculate and echo the duration
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        echo "Time taken for feature: $feature, target: $target - ${hours}h ${minutes}m"

    done
done