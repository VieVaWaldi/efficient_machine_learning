#!/usr/bin/env bash

#SBATCH --job-name=unet_distributed
#SBATCH --output=unet_distributed_%j.output
#SBATCH -p large
#SBATCH -N 80
#SBATCH --cpus-per-task=12
#SBATCH --time=03:45:00
#SBATCH --mail-type=all

# Varianten die Funktionieren:
# batchsize_Train = 8
# N 4   
# p long
# oder
# batchsize_Train = 16
# N 8
# p long
# oder 
# -p large
# -N 80


echo "submit host:"
echo $SLURM_SUBMIT_HOST
echo "submit dir:"
echo $SLURM_SUBMIT_DIR
echo "nodelist:"
echo $SLURM_JOB_NODELIST

#module load anaconda
#source /lustre/software/Anaconda3/x86_64/etc/profile.d/conda.sh
#conda activate /lustre/projects/breuer-group/conda_abreuer/pytorch_milan


cd $HOME/unet_seismic_new/
module load pytorch/arm22/1.10
#module load nvidia/cuda11.0/nvhpc/21.5


OMP_NUM_THREADS=12 mpiexec -n 80 --bind-to socket --report-bindings python unet_seismic_parallel.py