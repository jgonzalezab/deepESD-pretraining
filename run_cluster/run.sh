#!/bin/bash

export ENV_NAME="pnacc-gpu"
export JOB2RUN="./job.sh"
export DIR_TO_MOUNT="/gpfs/projects/meteo/WORK/gonzabad"

# Set the variable
export VAR_TARGET="pr"

export JOB_NAME="${VAR_TARGET}"
SLURM_PARAMS="--job-name=${JOB_NAME} --partition wngpu --time 72:00:00 --mem-per-cpu 127000 --output=./log/${JOB_NAME}.out --error=./log/${JOB_NAME}.err"
sbatch $SLURM_PARAMS $JOB2RUN