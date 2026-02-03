#!/bin/bash
#SBATCH -J FeHe_G
#SBATCH -A ISAAC-UTK0250
#SBATCH --partition=condo-camm
#SBATCH --qos=condo
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=48
#SBATCH --ntasks=16
#SBATCH --nodelist=ber[1434,1436,1438,1440]
#SBATCH --time=24:00:00
##SBATCH -e Demonstration.e%j
##SBATCH -o Demonstration.o%j
echo $SLURM_JOB_NODELIST
echo $SLURM_NTASKS

export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR
module purge
module load anaconda3
source $ANACONDA3_SH
conda activate /lustre/isaac24/scratch/tliang7/myVenvs/myvenv

srun -n 16 python run_seakmc_p.py

