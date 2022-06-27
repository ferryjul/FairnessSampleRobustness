#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --array=0-11
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=6G


srun -n 100 python3 default_credit_max_FPR_and_min_TPR.py --expe=$SLURM_ARRAY_TASK_ID
