import os
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn import image

from netneurotools import datasets as nntdata
from neuromaps.datasets import fetch_fslr
from surfplot import Plot

import matplotlib.pyplot as plt

def create_nifti_from_loadings(
    parcellation_dir,
    cat_parcellations,
    loadings,
    ):
    """Returns nifti from loadings/ weights (as dataframe)
    
    Parameters
    ----------
    parcellation_dir : str
        Path to parcellation directory where the parcellation nifti files and lookup tables are stored.
    cat_parcellations : list
        List of parcellations to be used in the analysis (e.g., ['Schaefer2018_200Parcels_17Networks_order', 'Tian_Subcortex_S2_7T', 'suit']).
        Together with lookup tables with first two columns as 'ROIid' and 'ROIabbr' and ';' delimiter.
    
    """
    # load atlases and combine them, remove overlapping voxels, and resample to space of first atlas
    for i, parcellation in enumerate(cat_parcellations):
        atlas = nib.load(os.path.join(parcellation_dir, f'{parcellation}.nii.gz'))
        if i == 0:
            combined_atlas = atlas
            first_atlas = atlas
            atlas_lut = pd.read_csv(os.path.join(parcellation_dir, f'{parcellation}.csv'), delimiter=';')
        else:
            # resample to first atlas space
            atlas = image.resample_to_img(atlas, first_atlas, interpolation='nearest')
            if parcellation == 'suit':
                # remove suit parcels > 28
                print('Removing `suit` parcels > 28')
                atlas = image.math_img('img1 * (img2 <= 28)', img1=atlas, img2=atlas)
            # update parcel number by adding 10000
            atlas = image.math_img(f'(img1 + {i*10000}) * (img2 > 0)', img1=atlas, img2=atlas)
            # remove voxels overlapping between atlas and combined_atlas
            atlas = image.math_img('img1 * (img2 == 0)', img1=atlas, img2=combined_atlas)
            # add atlas to combined_atlas
            combined_atlas = image.math_img('img1 + img2', img1=combined_atlas, img2=atlas)

            # load lookup table and update ROIid
            lut_ = pd.read_csv(os.path.join(parcellation_dir, f'{parcellation}.csv'), delimiter=';')
            lut_['ROIid'] = lut_['ROIid'] + i*10000
            atlas_lut = pd.concat([atlas_lut, lut_])

    roi_rename = pd.Series(atlas_lut['ROIid'].values,index=atlas_lut['ROIabbr']).to_dict()
    # check if any loadings not in atlas
    no_parcels = [x for x in loadings.columns.to_list() if x not in atlas_lut['ROIabbr'].values]
    if no_parcels:
        print(f'Warning: {no_parcels} not in atlas')
        loadings = loadings.drop(labels=no_parcels,axis=1)

    # check if any atlas not in loadings
    no_parcels = [x for x in atlas_lut['ROIabbr'].values if x not in loadings.columns.to_list()]
    if no_parcels:
        print(f'Warning: {no_parcels} not in loadings')

    # rename columns
    transdict = loadings.rename(columns=roi_rename).to_dict('list')

    # replace atlas values with weights
    values = combined_atlas.get_fdata()
    # Iterate over each element of the numpy array and replace ROI IDs with their values
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            for k in range(values.shape[2]):
                roi_id = values[i, j, k]
                if roi_id in transdict:
                    # take mean of list if multiple values
                    values[i, j, k] = np.mean(transdict[roi_id])
                else:
                    values[i, j, k] = 0        

    gmv_ = nib.Nifti1Image(dataobj=values, affine=combined_atlas.affine, header=combined_atlas.header)
    return gmv_


def plot_schaefer_surf(
        weights,
        parcellation_dir,
        scale=200,
        cmap='RdBu_r',
        output_dir=None,
        plt_show=False,
        color_range_by_largest_val=True,
        color_vmax_float=None):
    schaefer_lut = pd.read_csv(os.path.join(parcellation_dir, f'Schaefer2018_{scale}Parcels_17Networks_order.csv'), delimiter=';')
    schaefer = nntdata.fetch_schaefer2018('fslr32k')[f'{scale}Parcels17Networks']
    atlas = nib.load(schaefer).get_fdata()[0]

    # load look up table & create dictionary of ROI IDs and weight values
    schaefer_lut = pd.read_csv(os.path.join(parcellation_dir, f'Schaefer2018_{scale}Parcels_17Networks_order.csv'), delimiter=';')
    roi_rename = pd.Series(schaefer_lut['ROIid'].values,index=schaefer_lut['ROIabbr']).to_dict()
    transdict_ = weights.mean()
    transdict = transdict_.rename(roi_rename).to_dict()
    # replace parcel ID in atlas with weight value
    for i in range(1, scale + 1):
        atlas = np.where(atlas == i, transdict[i], atlas)
    
    if color_range_by_largest_val:
        largest_val = np.max(np.abs(atlas))
    else:
        largest_val = color_vmax_float
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']

    # generate plot
    p = Plot(surf_lh=lh, surf_rh=rh, size=(1600, 400), layout='row', zoom=1.2, mirror_views=True)
    #p = Plot(lh, rh)
    p.add_layer(atlas, cmap=cmap, color_range=(-largest_val, largest_val))

    kws = dict(location='bottom', draw_border=False, aspect=10, shrink=3.,
            decimals=1, pad=0)
    fig = p.build(cbar_kws=kws)
    ax = fig.get_axes()
    ax[1].tick_params(labelsize=14)
    if output_dir:
        fig.savefig(os.path.join(output_dir, f'schaefer2018_{scale}Parcels_projection.png'), dpi=300)
    if plt_show:
        plt.show()
    plt.close()
    return 0
