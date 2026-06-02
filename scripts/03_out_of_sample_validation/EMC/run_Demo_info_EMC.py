import os
from pathlib import Path


# Function to run the Python script using Spyder's runfile() method
def run_script_in_spyder(script_path, log_file):
    with open(log_file, 'a') as log:
        # Write a header for better log readability
        log.write(f"\n\n=== Running script for demographics description ===\n")
        runfile(script_path)


# Main execution
def main():
    # Define the path to the Python script and log directory
    script_path = r'C:\path\to\Demo_info_EMC.py'  # Update the path to your script
    log_dir = r'C:\path\to\log_directory'  # Update the path to your log directory

    # Create the log directory if it does not exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    run_script_in_spyder(script_path, log_file)


if __name__ == '__main__':
    main()
