# Estimate mean volumes and mean surface values inside ROI
/data/project/sleep_ENIGMA_insomnia/Tools/CAT12.8.1_r2040_R2017b_MCR_Linux/standalone/cat_standalone.sh -m /data/group/sysmed/TOOLS/MCR/v93 \
  -b /data/project/sleep_ENIGMA_insomnia/Tools/CAT12.8.1_r2040_R2017b_MCR_Linux/standalone/cat_standalone_get_ROI_values.m \
  /data/project/sleep_ENIGMA_insomnia/Data/Seoul/Raw/CAT12.8.1/label/catROI*.xml  -a1 "'ROI'"
