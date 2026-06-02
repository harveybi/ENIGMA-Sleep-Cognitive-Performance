# %%
import os
import subprocess
import shutil

# %%

con_sub_path = '/data/project/sleep_ENIGMA_insomnia/KUMS_Iran_20.07.2021/KUMS/recon-all_control'
pat_sub_path = '/data/project/sleep_ENIGMA_insomnia/KUMS_Iran_20.07.2021/KUMS/recon-all_patient'

# %%
print('\nProjecting started')
os.environ['SUBJECTS_DIR'] = con_sub_path

subject_name = 'sub-01'
gcs_file_dir = '/data/project/sleep_ENIGMA_insomnia/Codes/Yeo_FreeSurfer5.3/gcs'

command1 = 'mris_ca_label -l $SUBJECTS_DIR/' + subject_name + '/label/lh.cortex.label ' \
           + subject_name + ' lh $SUBJECTS_DIR/' + subject_name + '/surf/lh.sphere.reg ' \
           + gcs_file_dir + '/lh.Schaefer2018_400Parcels_7Networks.gcs ' \
                            '$SUBJECTS_DIR/' + subject_name + '/label/lh.Schaefer2018_400Parcels_7Networks_order.annot'

command2 = 'mris_ca_label -l $SUBJECTS_DIR/' + subject_name + '/label/rh.cortex.label ' \
           + subject_name + ' rh $SUBJECTS_DIR/' + subject_name + '/surf/rh.sphere.reg ' \
           + gcs_file_dir + '/rh.Schaefer2018_400Parcels_7Networks.gcs ' \
                            '$SUBJECTS_DIR/' + subject_name + '/label/rh.Schaefer2018_400Parcels_7Networks_order.annot'

os.system(command1)
os.system(command2)

print('\nProjecting finished')
# %%
"""
Compute atlas-based statistics.

# mris_anatomical_stats -a subjid/label/lh.aparc.annot -b subjid lh
"""

print('\nComputing started')

command3 = 'mris_anatomical_stats -a $SUBJECTS_DIR/' + subject_name + '/label/lh.Schaefer2018_400Parcels_7Networks_order.annot ' \
           '-f ' + '$SUBJECTS_DIR/' + subject_name + '/stats/lh.Schaefer2018_400Parcels_7Networks_order.stats ' \
           + subject_name + ' lh'

command4 = 'mris_anatomical_stats -a $SUBJECTS_DIR/' + subject_name + '/label/rh.Schaefer2018_400Parcels_7Networks_order.annot ' \
           '-f ' + '$SUBJECTS_DIR/' + subject_name + '/stats/rh.Schaefer2018_400Parcels_7Networks_order.stats ' \
           + subject_name + ' rh'

os.system(command3)
os.system(command4)

print('\nComputing finished')