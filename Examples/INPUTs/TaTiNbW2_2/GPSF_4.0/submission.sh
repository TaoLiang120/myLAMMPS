#!/bin/bash
#SBATCH -J Ta32Ti
#SBATCH -A ISAAC-UTK0250
#SBATCH --partition=condo-camm
#SBATCH --qos=condo
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=48
#SBATCH --ntasks=48
#SBATCH --nodelist=ber[1434,1436,1438,1440]
#SBATCH --time=96:00:00
##SBATCH -e Demonstration.e%j
##SBATCH -o Demonstration.o%j

echo $SLURM_JOB_NODELIST
echo $SLURM_NTASKS
ulimit -s unlimited
module load vasp/5.4.4

srun -n 48 vasp_std
