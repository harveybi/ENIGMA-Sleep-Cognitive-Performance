# %%
from xml.dom.minidom import parse
import xml.dom.minidom

from os import listdir
from os.path import isfile, join
import pandas as pd
import re

# %%
script_path = '/data/project/sleep_ENIGMA_insomnia/Codes/data_extract_scripts/'
save_path = '/data/project/sleep_ENIGMA_insomnia/Data/Seoul/Raw/'

# load csv ROI file, ROI_Schaefer2018_400Parcels_17Networks_order_Vgm.csv
roi_df = pd.read_csv(script_path + 'ROI_Schaefer2018_400Parcels_17Networks_order_Vgm.csv')

# %%
'''
Get the NCR ICR IQR TIV GM WM CSF WMH TSA from the XML file for each subject inside control_save_path and patient_save_path
Save the results in a dataframe, first column is 'Subject_ID' which is the same for both control and patient, second column is 'NCR',
third column is 'ICR', fourth column is 'IQR', fifth column is 'TIV', sixth column is 'GM', seventh column is 'WM', eighth column is 'CSF', ninth column is 'WMH', tenth column is 'TSA'

Make the final dataframe with all subjects and their covariates by for loop.
'''
def get_covariates(save_path):
    # get the list of subjects in the save_path
    subjects = [f for f in listdir(save_path) if isfile(join(save_path, f))]
    subjects.sort()

    # create a dataframe to save the results
    df = pd.DataFrame(columns=['Subject_ID', 'NCR', 'ICR', 'IQR', 'TIV', 'GM', 'WM', 'CSF', 'WMH', 'TSA'])
    # for each subject, get the covariates and save them in the dataframe
    for subject in subjects:
        subject_name = subject.split('.')[0]
        # get the xml file
        xml_file = save_path + 'CAT12.8.1/report/' + 'cat_' + subject_name + '.xml'
        # parse the xml file
        dom = xml.dom.minidom.parse(xml_file)
        # CGW
        cgw = dom.getElementsByTagName('vol_abs_CGW')
        cgw_data = cgw[0].firstChild.data
        aaa = re.findall(r"\d+\.?\d*", cgw_data)
        CSF = aaa[0]
        GM = aaa[1]
        WM = aaa[2]

        # NCR
        ncr_get = dom.getElementsByTagName('NCR')
        NCR = ncr_get[1].firstChild.data

        # ICR
        icr_get = dom.getElementsByTagName('ICR')
        ICR = icr_get[1].firstChild.data

        # IQR
        iqr_get = dom.getElementsByTagName('IQR')
        IQR = iqr_get[0].firstChild.data

        # TIV
        tiv_get = dom.getElementsByTagName('vol_TIV')
        TIV = tiv_get[1].firstChild.data

        # WMH
        wmh_get = dom.getElementsByTagName('vol_abs_WMH')
        WMH = wmh_get[0].firstChild.data

        # TSA
        if dom.getElementsByTagName('surf_TSA') is not None:
            tsa_get = dom.getElementsByTagName('surf_TSA')
            TSA = tsa_get[1].firstChild.data

        # save the covariates in the dataframe
        df = df.append({'Subject_ID': subject_name, 'NCR': NCR, 'ICR': ICR, 'IQR': IQR, 'TIV': TIV, 'GM': GM, 'WM': WM, 'CSF': CSF, 'WMH': WMH, 'TSA': TSA}, ignore_index=True)
    return df

# %%
covariates_df = get_covariates(save_path)

# %%
# get the roi_df without the first column 'names'
roi_df_need = roi_df.iloc[:, 1:]

# append roi_df to covariates_df, but for roi_df we not need the first column 'name'
covariates_df_all = pd.concat([covariates_df, roi_df_need], axis=1)

# %%
# save the dataframe as csv file
covariates_df_all.to_csv(script_path + 'Seoul_cat12.8.1_1surf_rois_Schaefer2018_400Parcels_17Networks_order.csv', index=False)
