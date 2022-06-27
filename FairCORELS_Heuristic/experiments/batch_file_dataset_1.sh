#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --array=0-146
#SBATCH --ntasks=75
#SBATCH --mem-per-cpu=8G


srun python bench_mpi_new.py --dataset=1 --epsilon=$SLURM_ARRAY_TASK_ID --debug=0
