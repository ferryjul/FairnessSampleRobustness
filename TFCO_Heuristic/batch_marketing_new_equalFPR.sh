#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=6G


srun -n 100 python3 marketing_equalFPR-preproc-correct.py --expe=$SLURM_ARRAY_TASK_ID
