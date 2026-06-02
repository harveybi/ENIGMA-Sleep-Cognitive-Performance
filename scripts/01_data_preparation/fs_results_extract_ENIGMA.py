# %%
import os
from os import listdir
import shutil

import pandas as pd

# %%
def individual_space_mapping(sub_path, gcs_path):
    """Project Schaefer2018 parcellation to individual space.

    mris_ca_label -l $SUBJECTS_DIR/<subject_name>/label/lh.cortex.label \
        <subject_name> lh $SUBJECTS_DIR/<subject_name>/surf/lh.sphere.reg \
        <gcs_file_dir>/lh.Schaefer2018_<N>Parcels_<7/17>Networks.gcs \
        $SUBJECTS_DIR/<subject_name>/label/lh.Schaefer2018_<N>Parcels_<7/17>Networks_order.annot

    mris_ca_label -l $SUBJECTS_DIR/<subject_name>/label/rh.cortex.label \
        <subject_name> rh $SUBJECTS_DIR/<subject_name>/surf/rh.sphere.reg \
        <gcs_file_dir>/rh.Schaefer2018_<N>Parcels_<7/17>Networks.gcs \
        $SUBJECTS_DIR/<subject_name>/label/rh.Schaefer2018_<N>Parcels_<7/17>Networks_order.annot

    Args:
        sub_path (str): Path to subject directory.
        gcs_path (str): Path to GCS file directory.

    Returns:
        Terminal will return the FreeSurfer running results.
    """

    print('\nProjecting started')
    os.environ['SUBJECTS_DIR'] = sub_path
    gcs_file_dir = gcs_path

    subjects = listdir(sub_path)
    subjects.sort()
    # sub_list = subjects[1:]  # Notice! This is used to remove the folder which is not a subject
    # only keep folders name start with 'sub-'
    sub_list = [sub for sub in subjects if sub.startswith('sub-')]

    for sub in sub_list:

        subject_name = sub
        print('\nProjecting: ' + subject_name)

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

        # print('\n' + command1)
        # print('\n' + command2)

    print('\nProjecting finished')


def stats_computing(sub_path):
    """Compute atlas-based statistics.

    mris_anatomical_stats -a subjid/label/lh.aparc.annot -b subjid lh

    Args:
        sub_path (str): Path to subject directory.

    Returns:
        Terminal will return the FreeSurfer running results.
    """

    print('\nStats computing started')
    os.environ['SUBJECTS_DIR'] = sub_path

    subjects = listdir(sub_path)
    subjects.sort()
    # sub_list = subjects[1:]  # Notice! This is used to remove the folder which is not a subject
    # only keep folders name start with 'sub-'
    sub_list = [sub for sub in subjects if sub.startswith('sub-')]

    for sub in sub_list:

        subject_name = sub
        print('\nComputing: ' + subject_name)

        command1 = 'mris_anatomical_stats -a $SUBJECTS_DIR/' + subject_name + '/label/lh.Schaefer2018_400Parcels_7Networks_order.annot ' \
                   '-f ' + '$SUBJECTS_DIR/' + subject_name + '/stats/lh.Schaefer2018_400Parcels_7Networks_order.stats ' \
                   + subject_name + ' lh'

        command2 = 'mris_anatomical_stats -a $SUBJECTS_DIR/' + subject_name + '/label/rh.Schaefer2018_400Parcels_7Networks_order.annot ' \
                   '-f ' + '$SUBJECTS_DIR/' + subject_name + '/stats/rh.Schaefer2018_400Parcels_7Networks_order.stats ' \
                   + subject_name + ' rh'

        os.system(command1)
        os.system(command2)

        print('\n' + subject_name + 'computing finished')

    print('\nStats computing finished')


def results_extract(sub_path, site_name, atlas_name):
    """Compute atlas-based statistics.

    Extract each subjects' cortical thickness and surface area in to txt file.

    Args:
        sub_path (str): Path to subject directory.
        site_name (str): Site name.
        atlas_name (str): Atlas name, like 'DK' or 'Schaefer'.

    Returns:
        Terminal will return the FreeSurfer running results.
        The results will be saved in this script's directory.
    """

    print('\nResults extracting started')
    os.environ['SUBJECTS_DIR'] = sub_path

    subjects = listdir(sub_path)
    subjects.sort()
    # sub_list = subjects[1:]  # Notice! This is used to remove the folder which is not a subject
    # only keep folders name start with 'sub-'
    sub_list = [sub for sub in subjects if sub.startswith('sub-')]
    sub_list_str = ' '.join(sub_list)

    if atlas_name == 'DK':
        parc_name = 'aparc '

        # lh_thickness
        command1 = 'aparcstats2table --hemi lh --meas thickness --parc ' + parc_name + \
                   '--tablefile ' + site_name + '_lh_thickness_' + atlas_name + '.txt --subjects ' + sub_list_str + ' --skip'  # + ' --debug'
        os.system(command1)
        # print('\n' + command1)

        # rh_thickness
        command2 = 'aparcstats2table --hemi rh --meas thickness --parc ' + parc_name + \
                   '--tablefile ' + site_name + '_rh_thickness_' + atlas_name + '.txt --subjects ' + sub_list_str + ' --skip'
        os.system(command2)
        # print('\n' + command2)

        # lh_area
        command3 = 'aparcstats2table --hemi lh --meas area --parc ' + parc_name + \
                   '--tablefile ' + site_name + '_lh_area_' + atlas_name + '.txt --subjects ' + sub_list_str + ' --skip'
        os.system(command3)
        # print('\n' + command3)

        # rh_area
        command4 = 'aparcstats2table --hemi rh --meas area --parc ' + parc_name + \
                   '--tablefile ' + site_name + '_rh_area_' + atlas_name + '.txt --subjects ' + sub_list_str + ' --skip'
        os.system(command4)
        # print('\n' + command4)

    elif atlas_name == 'Schaefer':
        parc_name = 'Schaefer2018_400Parcels_7Networks_order '

        # lh_thickness
        command1 = 'aparcstats2table --hemi lh --meas thickness --parc ' + parc_name + \
                   '--tablefile ' + site_name + '_lh_thickness_' + atlas_name + '.txt --subjects ' + sub_list_str + ' --skip'
        os.system(command1)

        # rh_thickness
        command2 = 'aparcstats2table --hemi rh --meas thickness --parc ' + parc_name + \
                   '--tablefile ' + site_name + '_rh_thickness_' + atlas_name + '.txt --subjects ' + sub_list_str + ' --skip'
        os.system(command2)

        # lh_area
        command3 = 'aparcstats2table --hemi lh --meas area --parc ' + parc_name + \
                   '--tablefile ' + site_name + '_lh_area_' + atlas_name + '.txt --subjects ' + sub_list_str + ' --skip'
        os.system(command3)

        # rh_area
        command4 = 'aparcstats2table --hemi rh --meas area --parc ' + parc_name + \
                   '--tablefile ' + site_name + '_rh_area_' + atlas_name + '.txt --subjects ' + sub_list_str + ' --skip'
        os.system(command4)

    print('\nResults extracting finished')


def csv_maker(file_path, site_name, atlas_name):
    """Combine the results of lh and rh into one csv file.

    Args:
        file_path (str): Path to the file.
        site_name (str): Name of the site.
        atlas_name (str): Name of the atlas.

    Returns:
        Saved csv file.
    """

    print('\nMaking csv file started')
    # thickness
    df_lh = pd.read_table(file_path + site_name + '_lh_thickness_' + atlas_name + '.txt', sep='\s+')
    df_rh = pd.read_table(file_path + site_name + '_rh_thickness_' + atlas_name + '.txt', sep='\s+')

    if atlas_name == 'DK':
        df_concat_lh_thickness = df_lh.drop(columns=['BrainSegVolNotVent', 'eTIV'])
        df_concat_rh_thickness = df_rh.drop(columns=['rh.aparc.thickness'])
        df_thickness = pd.concat([df_concat_lh_thickness, df_concat_rh_thickness], axis=1)
        df_thickness = df_thickness.rename(columns={'lh.aparc.thickness': 'subject_ID'})
        df_thickness.to_csv(file_path + site_name + '_thickness_' + atlas_name + '.csv', index=False)

    elif atlas_name == 'Schaefer':
        df_concat_lh_thickness = df_lh.drop(columns=['BrainSegVolNotVent', 'eTIV'])
        df_concat_rh_thickness = df_rh.drop(columns=['rh.Schaefer2018_400Parcels_7Networks_order.thickness'])
        df_thickness = pd.concat([df_concat_lh_thickness, df_concat_rh_thickness], axis=1)
        df_thickness = df_thickness.rename(columns={'lh.Schaefer2018_400Parcels_7Networks_order.thickness': 'subject_ID'})
        df_thickness.to_csv(file_path + site_name + '_thickness_' + atlas_name + '.csv', index=False)

    # area
    df_lh = pd.read_table(file_path + site_name + '_lh_area_' + atlas_name + '.txt', sep='\s+')
    df_rh = pd.read_table(file_path + site_name + '_rh_area_' + atlas_name + '.txt', sep='\s+')

    if atlas_name == 'DK':
        df_concat_lh_area = df_lh.drop(columns=['BrainSegVolNotVent', 'eTIV'])
        df_concat_rh_area = df_rh.drop(columns=['rh.aparc.area'])
        df_area = pd.concat([df_concat_lh_area, df_concat_rh_area], axis=1)
        df_area = df_area.rename(columns={'lh.aparc.area': 'subject_ID'})
        df_area.to_csv(file_path + site_name + '_area_' + atlas_name + '.csv', index=False)

    elif atlas_name == 'Schaefer':
        df_concat_lh_area = df_lh.drop(columns=['BrainSegVolNotVent', 'eTIV'])
        df_concat_rh_area = df_rh.drop(columns=['rh.Schaefer2018_400Parcels_7Networks_order.area'])
        df_area = pd.concat([df_concat_lh_area, df_concat_rh_area], axis=1)
        df_area = df_area.rename(columns={'lh.Schaefer2018_400Parcels_7Networks_order.area': 'subject_ID'})
        df_area.to_csv(file_path + site_name + '_area_' + atlas_name + '.csv', index=False)

    print('\nMaking csv file finished')


def subcortical_extract(file_path, sub_path, site_name):
    """Generate .txt file of FreeSurfer subcortical stats.

    asegstats2table --subjects bert ernie fred margaret --meas volume --tablefile aseg_stats.txt

    Args:
        file_path (str): Path to the script.
        sub_path (str): Path to the subject folder.
        site_name (str): Name of the site.

    Returns:
        Saved .txt and .csv file. The results will be saved in this script's directory.
    """

    print('\nSubcortical stats extracting started')
    os.environ['SUBJECTS_DIR'] = sub_path

    subjects = listdir(sub_path)
    subjects.sort()
    # sub_list = subjects[1:]  # Notice! This is used to remove the folder which is not a subject
    # only keep folders name start with 'sub-'
    sub_list = [sub for sub in subjects if sub.startswith('sub-')]
    sub_list_str = ' '.join(sub_list)

    command = 'asegstats2table --subjects ' + sub_list_str + ' --meas volume --tablefile ' + site_name + '_aseg_stats.txt' + ' --skip'
    os.system(command)

    # make csv file
    df = pd.read_table(file_path + site_name + '_aseg_stats.txt', sep='\s+')
    df = df.rename(columns={'Measure:volume': 'subject_ID'})

    df.to_csv(file_path + site_name + '_subcortical_volume.csv', index=False)

    print('\nSubcortical stats extracting finished')


# %%
"""
Usage example
This example is based on KUMS (Kermanshah University of Medical Sciences) data, which has controlbids and patientbids two folders.
So the control and patient here will be the site names like KI, Liege, etc.

After you run the recon-all command on your subjects, you will have a folder named 'recon-all_***' which includes a 'fsaverage'
folder and all subjects' folders. The subjects folder includes all the results of FreeSurfer for that subject.
"""
# So here, you should change the path to your 'recon-all_***' folder
subject_path = '/data/project/sleep_ENIGMA_harmonization/data/Juelich_FreeSurfer'
# Your gcs file path
gcs_file_path = '/data/project/sleep_ENIGMA_insomnia/Codes/Yeo_FreeSurfer5.3/gcs'
# Your script path. Notice, the output of this script will be saved in this script's directory.
save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Code/'

# %%
"""First step, individual space mapping for Schaefer atlas"""
# individual_space_mapping(subject_path, gcs_file_path)
# stats_computing(subject_path)
#
# # %%
# """Second step, extract cortical thickness and surface area for Schaefer and DK atlas"""
# # Schaefer atlas
# results_extract(subject_path, 'Juelich', 'Schaefer')
#
# # DK atlas
# results_extract(subject_path, 'Juelich', 'DK')

# %%
# make csv files
# Schaefer atlas
csv_maker(save_path, 'Juelich', 'Schaefer')

# DK atlas
csv_maker(save_path, 'Juelich', 'DK')

# %%
"""Third step, extract subcortical volume"""
# subcortical extract
subcortical_extract(save_path, subject_path, 'Juelich')
