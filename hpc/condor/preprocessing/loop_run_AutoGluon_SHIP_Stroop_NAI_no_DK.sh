#!/usr/bin/env bash
set -euo pipefail

echo ">>> RUN: Brain × Stroop"
bash run_AutoGluon_SHIP_Stroop_NAI_no_DK.sh Brain Stroop 20 0

echo ">>> RUN: Sleep_Cov_Brain_Shuffle × Stroop"
bash run_AutoGluon_SHIP_Stroop_NAI_no_DK.sh Sleep_Cov_Brain_Shuffle Stroop 20 0

echo ">>> RUN: Brain × Memory"
bash run_AutoGluon_SHIP_Stroop_NAI_no_DK.sh Brain Memory 20 0

echo ">>> RUN: Sleep_Cov_Brain_Shuffle × Memory"
bash run_AutoGluon_SHIP_Stroop_NAI_no_DK.sh Sleep_Cov_Brain_Shuffle Memory 20 0

echo ">>> RUN: Sleep_Cov_Brain × Memory_rgo_age"
bash run_AutoGluon_SHIP_Stroop_NAI_no_DK.sh Sleep_Cov_Brain_Shuffle Memory_rgo_age 20 0

