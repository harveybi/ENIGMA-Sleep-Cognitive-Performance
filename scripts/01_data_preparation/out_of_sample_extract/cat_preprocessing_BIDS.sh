# Preprocessing
/data/group/sysmed/TOOLS/CAT12.8.1_r2040_R2017b_MCR_Linux_HB/standalone/cat_standalone.sh -m /data/group/sysmed/TOOLS/MCR/v93 \
 -b /data/group/sysmed/TOOLS/CAT12.8.1_r2040_R2017b_MCR_Linux_HB/standalone/cat_standalone_segment_enigma.m \
  /data/project/sleep_ENIGMA_insomnia/Data/KUMS_Iran_20.07.2021/KUMS/controlbids/CAT_test/sub*/anat/sub*.nii.gz \
  -a "matlabbatch{1}.spm.tools.cat.estwrite.output.surface = 0;"