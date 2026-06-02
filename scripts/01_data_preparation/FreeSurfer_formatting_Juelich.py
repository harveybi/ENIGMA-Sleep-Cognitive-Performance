import os

# %%
sleepless_dir = '/data/project/sleep_ENIGMA_harmonization/data/sleepless/'
somnosafe_dir = '/data/project/sleep_ENIGMA_harmonization/data/somnosafe/'

# %%
sleepless_subjects = os.listdir(sleepless_dir)
somnosafe_subjects = os.listdir(somnosafe_dir)

# %%
# only keep the folders starting with 'sub-'
sleepless_subjects = [s for s in sleepless_subjects if s.startswith('sub-')]
somnosafe_subjects = [s for s in somnosafe_subjects if s.startswith('sub-')]

# %%
freesurfer_dir = '/data/project/sleep_ENIGMA_harmonization/data/Juelich_FreeSurfer/'

# %%
# for each subject in sleepless and somnosafe, copy the .nii.gz files to the FreeSurfer directory. .nii.gz files are like /data/project/sleep_ENIGMA_harmonization/data/somnosafe/sub-105006/ses-bl/anat/sub-105006_ses-bl_T1w.nii.gz
for subject in sleepless_subjects:
    subject_dir = os.path.join(sleepless_dir, subject, 'ses-bl', 'anat')
    T1w_file = os.path.join(subject_dir, f'{subject}_ses-bl_T1w.nii.gz')
    if not os.path.exists(subject_dir):
        print(f'{subject_dir} folder does not exist')
    else:
        if not os.path.exists(T1w_file):
            T1w_file = os.path.join(subject_dir, f'{subject}_ses-bl_run-01_T1w.nii.gz')
        os.system(f'cp {T1w_file} {freesurfer_dir}')
    print(f'{T1w_file} copied to {freesurfer_dir}')


for subject in somnosafe_subjects:
    subject_dir = os.path.join(somnosafe_dir, subject, 'ses-bl', 'anat')
    T1w_file = os.path.join(subject_dir, f'{subject}_ses-bl_T1w.nii.gz')
    if not os.path.exists(subject_dir):
        print(f'{subject_dir} folder does not exist')
    else:
        if not os.path.exists(T1w_file):
            T1w_file = os.path.join(subject_dir, f'{subject}_ses-bl_run-01_T1w.nii.gz')
        os.system(f'cp {T1w_file} {freesurfer_dir}')
    print(f'{T1w_file} copied to {freesurfer_dir}')

# %%
# check whether the files are copied correctly
target_list = os.listdir(freesurfer_dir)
# sort the list
target_list.sort()
print(target_list)

# %%
# for the files name include run-01, rename them to the format like sub-105006_ses-bl_T1w.nii.gz
for target in target_list:
    if 'run-01' in target:
        new_target = target.replace('run-01_', '')
        os.rename(os.path.join(freesurfer_dir, target), os.path.join(freesurfer_dir, new_target))
        print(f'{target} renamed to {new_target}')

# %%
target_list = os.listdir(freesurfer_dir)
# sort the list
target_list.sort()
print(target_list)

# %%
# for the files, remove the prefix 'sub-'
for target in target_list:
    new_target = target.replace('sub-', '')
    os.rename(os.path.join(freesurfer_dir, target), os.path.join(freesurfer_dir, new_target))
    print(f'{target} renamed to {new_target}')
