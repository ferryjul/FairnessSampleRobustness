#!/bin/bash

echo "Scripts launcher for all datasets"

if [ "$#" -eq 0 ]; then
    echo "Launching scripts for all 4 datasets for all metrics"
    sbatch batch_file_dataset_1.sh
    sbatch batch_file_dataset_2.sh
    sbatch batch_file_dataset_3.sh
    sbatch batch_file_dataset_4.sh
fi

if [ "$#" -eq 1 ]; then
    if [ "$1" -eq 1 ]; then
        echo "  Launching scripts only for dataset 1 for all metrics"
	    sbatch batch_file_dataset_1.sh
    fi

    if [ "$1" -eq 2 ]; then
        echo "  Launching scripts only for dataset 2 for all metrics"
	    sbatch batch_file_dataset_2.sh
    fi

    if [ "$1" -eq 3 ]; then
	    echo "  Launching scripts only for dataset 3 for all metrics"
	    sbatch batch_file_dataset_3.sh
    fi

    if [ "$1" -eq 4 ]; then
	    echo "  Launching scripts only for dataset 4 for all metrics"
	    sbatch batch_file_dataset_4.sh
    fi

fi
