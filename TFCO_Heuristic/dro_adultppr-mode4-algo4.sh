#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=02:00:00
#SBATCH --job-name=droSTA
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray

python3 adult_ppr0.8.py --drovalseed=$SLURM_ARRAY_TASK_ID --mode=4 --algo=4