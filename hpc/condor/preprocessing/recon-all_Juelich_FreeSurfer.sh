#!/bin/bash
source /etc/profile.d/modules.sh

export SUBJECTS_DIR=/data/project/sleep_ENIGMA_harmonization/data/Juelich_FreeSurfer

module avail freesurfer

module load freesurfer/7.4

# Check if the input argument is provided
if [ -z "$1" ]; then
    echo "Error: No subject ID provided."
    exit 1
fi

recon-all -s sub-"$1" -i /data/project/sleep_ENIGMA_harmonization/data/Juelich_FreeSurfer/$1_ses-bl_T1w.nii.gz -all >> "/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/combined_logs/Juelich_FreeSurfer/${1}_log.txt" 2>&1