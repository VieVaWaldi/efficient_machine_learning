#!/usr/bin/env bash
echo "Dont forget to schedule a milan gpu"
module load anaconda
source /lustre/software/Anaconda3/x86_64/etc/profile.d/conda.sh
conda activate /lustre/projects/breuer-group/conda_abreuer/pytorch_milan
module load nvidia/cuda11.0/nvhpc/21.5

