import os
from pathlib import Path


# Function to run the Python script using Spyder's runfile() method
def run_script_in_spyder(script_path, feature, target, log_file):
    with open(log_file, 'a') as log:
        # Prepare the arguments to be passed to the script
        args = f"{feature} {target}"

        # Write a header for better log readability
        log.write(f"\n\n=== Running script for feature: {feature}, target: {target} ===\n")
        runfile(script_path, args=args)



# Main execution
def main():
    # Define the path to the Python script and log directory
    script_path = r'C:\path\to\XGBoost_EMC_validate_Stroop.py'  # Update the path to your script
    log_dir = r'C:\path\to\log_directory'  # Update the path to your log directory

    # Create the log directory if it does not exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Define the combinations of features and the target variable
    feature_combines = ['Sleep', 'Cov', 'Sleep_Cov', 'Sleep_Shuffle_Cov',
                        'Brain', 'Sleep_Cov_Brain', 'Sleep_Cov_Brain_Shuffle',
                        'Sleep_APOE', 'Sleep_APOE_Shuffle', 'Sleep_Cov_APOE', 'Sleep_Cov_APOE_Shuffle',
                        'Sleep_Cov_Brain_APOE', 'Sleep_Cov_Brain_APOE_Shuffle']
    targets = ['Stroop', 'Stroop_rgo_age']

    # Loop over the feature combinations and run the script
    for feature in feature_combines:
        for target in targets:
            if '_rgo_age' in target and feature not in ['Sleep_Cov', 'Sleep_Cov_Brain']:
                continue  # Skip this iteration if the feature is not allowed
            log_file = os.path.join(log_dir, f'{target}_{feature}_log.txt')
            print(f'Running script for feature: {feature}, target: {target}')
            run_script_in_spyder(script_path, feature, target, log_file)


if __name__ == '__main__':
    main()
