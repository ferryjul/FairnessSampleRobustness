#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --array=0-63
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=10G


srun -n 5 python3 sample_robustness_with_validation_mpi.py --expe=$SLURM_ARRAY_TASK_ID
