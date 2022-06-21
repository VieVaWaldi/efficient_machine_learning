#!/usr/bin/env bash

#SBATCH --job-name=mlp_distributed
#SBATCH --output=mlp_distributed_%j.output
#SBATCH -p long
#SBATCH -N 4
#SBATCH --cpus-per-task=12
#SBATCH --time=01:30:00
#SBATCH --mail-type=all
#SBATCH --mail-user=walter.ehrenberger@uni-jena.de

echo "submit host:"
echo $SLURM_SUBMIT_HOST
echo "submit dir:"
echo $SLURM_SUBMIT_DIR
echo "nodelist:"
echo $SLURM_JOB_NODELIST

cd $HOME/code/efficient_machine_learning/7th_chapter/mlp_fashion_mnist
module load pytorch/arm22/1.10

OMP_NUM_THREADS=12 mpiexec -n 4 --bind-to socket --report-bindings python mlp_fashion_mnist.py