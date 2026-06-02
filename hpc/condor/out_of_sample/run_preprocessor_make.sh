#!/bin/bash
source /home/h.bi/anaconda3/etc/profile.d/conda.sh  # Change the path to your conda.sh file

conda deactivate
conda activate XGBoost

# Define the path to your script and log directory
script_path='/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/preprocessor_make_save.py'  # Change the path to your script
log_path='/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/combined_logs/preprocessor_make/'  # Change the path to your log directory

# Create the log directory if it does not exist
mkdir -p "$log_path"

# Define the combinations of features and the target variable
feature_combines=('Sleep' 'Cov' 'Sleep_Cov' 'Sleep_Shuffle_Cov' 'Brain' 'Sleep_Cov_Brain' 'Sleep_Cov_Brain_Shuffle')  # 'Sleep_APOE' 'Sleep_APOE_Shuffle' 'Sleep_Cov_APOE' 'Sleep_Cov_APOE_Shuffle' 'Sleep_Cov_Brain_APOE' 'Sleep_Cov_Brain_APOE_Shuffle'
target='Stroop'

# Loop over each feature combination to run the script and log outputs
for arg in "${feature_combines[@]}"; do
    log_file="${log_path}${arg}_${target}_log.txt"
    python3 "$script_path" "$arg" "$target" >> "$log_file" 2>&1
done
