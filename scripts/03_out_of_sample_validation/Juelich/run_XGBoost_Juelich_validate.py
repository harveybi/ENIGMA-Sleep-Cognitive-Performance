import os
from pathlib import Path
import subprocess


# Function to run the Python script using subprocess
def run_script(script_path, feature, target, sess, log_file):
    with open(log_file, 'a') as log:  # Open the log file in append mode
        cmd = ["python3", script_path, feature, target, sess]

        # Write a header for better log readability
        log.write(f"\n\n=== Running script for session: {sess} feature: {feature}, target: {target} ===\n")

        # Run the command and capture the output in the log file
        try:
            subprocess.run(cmd, stdout=log, stderr=log, check=True)
        except subprocess.CalledProcessError as e:
            log.write(f"\nError occurred while running: {cmd}\n")
            log.write(str(e) + "\n")


# Main execution
def main():
    # Define the path to the Python script and log directory
    script_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/Juelich/XGBoost_Juelich_validate.py'  # Update the path to your script
    log_dir = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/Out-of-sample_validation/combined_logs/Juelich/XGBoost'  # Update the path to your log directory

    # Create the log directory if it does not exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Define the combinations of features and the target variable
    feature_combines = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
                        'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
                        'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle',
                        'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle']
    targets = ['Memory_Letter', 'Memory_Spatial', 'Memory_Letter_rgo_age', 'Memory_Spatial_rgo_age']
    sessions = ['sess-1', 'sess-2', 'sess-SD']

    # Loop over the feature combinations and run the script
    for sess in sessions:
        for feature in feature_combines:
            for target in targets:
                # Check condition: if the target contains '_rgo_age', limit the features
                if '_rgo_age' in target and feature not in ['Sleep_Cov', 'Sleep_Cov_Brain']:
                    continue  # Skip this iteration if the feature is not allowed

                # Define the log file path
                log_file = os.path.join(log_dir, f'{sess}_{target}_{feature}_log.txt')

                # Print execution info
                print(f'Running script for session: {sess}, feature: {feature}, target: {target}')

                # Run the script
                run_script(script_path, feature, target, sess, log_file)


if __name__ == '__main__':
    main()
