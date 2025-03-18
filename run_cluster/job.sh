#!/bin/bash

export PATH=$PATH:/bin/

source /gpfs/projects/meteo/WORK/gonzabad/miniforge3/etc/profile.d/conda.sh
conda activate $ENV_NAME

cd /gpfs/projects/meteo/WORK/gonzabad/deepESD-pretraining/scripts

# # Train and compute projections for the original model
# python -u original.py $VAR_TARGET

# # Finetune
# export NUM_ENSEMBLES=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
# for member in "${NUM_ENSEMBLES[@]}"
# do
#     echo $member  
#     python -u finetune.py $VAR_TARGET $member original
#     python -u finetune.py $VAR_TARGET $member pretrained
#     python -u finetune.py $VAR_TARGET $member pretrained_finetuning
# done

# Compute XAI metrics (for a single member of the ensemble)
python -u xai_original.py $VAR_TARGET

python -u xai_finetune.py $VAR_TARGET original
python -u xai_finetune.py $VAR_TARGET pretrained
python -u xai_finetune.py $VAR_TARGET pretrained_finetuning