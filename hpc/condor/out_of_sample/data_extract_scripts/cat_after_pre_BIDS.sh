# Estimate mean volumes and mean surface values inside ROI
/data/project/sleep_ENIGMA_insomnia/Tools/CAT12.8.1_r2040_R2017b_MCR_Linux/standalone/cat_standalone.sh -m /data/group/sysmed/TOOLS/MCR/v93 \
  -b /data/project/sleep_ENIGMA_insomnia/Tools/CAT12.8.1_r2040_R2017b_MCR_Linux/standalone/cat_standalone_get_ROI_values.m \
  /data/project/sleep_ENIGMA_insomnia/Data/KUMS_Iran_20.07.2021/KUMS/controlbids/CAT_test/CAT12.8.1/sub*/anat/catROI*.xml  -a1 "'ROI'"