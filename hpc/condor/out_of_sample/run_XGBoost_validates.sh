#!/bin/bash
source /home/h.bi/anaconda3/etc/profile.d/conda.sh  # Change the path to your conda.sh file

conda deactivate
conda activate XGBoost

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: No argument supplied. Please specify the validation folder (e.g., Liege)."
    exit 1
fi

# Define the script path
validation_folder="$1"
script_path="/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/${validation_folder}/run_XGBoost_${validation_folder}_validate.py"

# Check if the Python script exists
if [ ! -f "$script_path" ]; then
    echo "Error: Script not found at $script_path"
    exit 1
fi

# Run the Python script
python3 "$script_path" >> "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/combined_logs/condor_XGBoost/XGBoost_${validation_folder}_validate.txt" 2>&1