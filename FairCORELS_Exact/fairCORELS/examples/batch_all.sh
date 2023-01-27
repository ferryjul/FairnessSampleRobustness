#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --array=0-63
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=5G


srun -n 5 python3 sample_robust_frontier_epsilon_expes_script_mpi.py --expe=$SLURM_ARRAY_TASK_ID
