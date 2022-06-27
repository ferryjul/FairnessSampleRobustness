#!/bin/bash

echo "Algorithm 4 scripts launcher"
if [ "$#" -eq 0 ]; then
    echo "Launching scripts for the 2 first expes"
    for DATASET in 1 2
    do
        if [ "$DATASET" -eq 1 ]; then
            echo "  Launching expes 1 : adult dataset, PPR"
			sbatch --array=0-99 dro_adultppr-mode0-algo4.sh
			sbatch --array=0-99 dro_adultppr-mode1-algo4.sh
			sbatch --array=0-99 dro_adultppr-mode2-algo4.sh
			sbatch --array=0-99 dro_adultppr-mode3-algo4.sh
			sbatch --array=0-99 dro_adultppr-mode4-algo4.sh
			sbatch --array=0-99 dro_adultppr-mode5-algo4.sh
        fi

        if [ "$DATASET" -eq 2 ]; then
            echo "  Launching expes 2 : COMPAS dataset, TPR"
			sbatch --array=0-99 dro_compastpr-mode0-algo4.sh
			sbatch --array=0-99 dro_compastpr-mode1-algo4.sh
			sbatch --array=0-99 dro_compastpr-mode2-algo4.sh
			sbatch --array=0-99 dro_compastpr-mode3-algo4.sh
			sbatch --array=0-99 dro_compastpr-mode4-algo4.sh
			sbatch --array=0-99 dro_compastpr-mode5-algo4.sh	
        fi

    done

fi

if [ "$#" -eq 1 ]; then
    if [ "$1" -eq 1 ]; then
        echo "  Launching ONLY expes 1 : adult dataset, PPR"
	    sbatch --array=0-99 dro_adultppr-mode0-algo4.sh
	    sbatch --array=0-99 dro_adultppr-mode1-algo4.sh
	    sbatch --array=0-99 dro_adultppr-mode2-algo4.sh
	    sbatch --array=0-99 dro_adultppr-mode3-algo4.sh
	    sbatch --array=0-99 dro_adultppr-mode4-algo4.sh
	    sbatch --array=0-99 dro_adultppr-mode5-algo4.sh
    fi

    if [ "$1" -eq 2 ]; then
        echo "  Launching ONLY expes 2 : COMPAS dataset, TPR"
	    sbatch --array=0-99 dro_compastpr-mode0-algo4.sh
	    sbatch --array=0-99 dro_compastpr-mode1-algo4.sh
	    sbatch --array=0-99 dro_compastpr-mode2-algo4.sh
	    sbatch --array=0-99 dro_compastpr-mode3-algo4.sh
	    sbatch --array=0-99 dro_compastpr-mode4-algo4.sh
	    sbatch --array=0-99 dro_compastpr-mode5-algo4.sh	
    fi

fi
