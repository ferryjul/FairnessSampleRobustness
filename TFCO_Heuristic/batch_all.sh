#!/bin/bash

echo "Scripts launcher for both Algorithm 3 & 4"

if [ "$#" -eq 0 ]; then
    echo "Launching scripts for all 4 expes for both algorithms"
    ./batch_all_algo3.sh
    ./batch_all_algo4.sh
    sbatch batch_marketing_new_equalFPR.sh
    sbatch batch_default_credit_min_tpr.sh
fi

if [ "$#" -eq 1 ]; then
    if [ "$1" -eq 1 ]; then
        echo "  Launching ONLY expes 1 : adult dataset, PPR"
	    ./batch_all_algo3.sh 1
	    ./batch_all_algo4.sh 1
    fi

    if [ "$1" -eq 2 ]; then
        echo "  Launching ONLY expes 2 : COMPAS dataset, TPR"
		./batch_all_algo3.sh 2
		./batch_all_algo4.sh 2
    fi

    if [ "$1" -eq 3 ]; then
	    echo "  Launching ONLY expes 3 : Bank Marketing dataset, FPR"
		sbatch batch_marketing_new_equalFPR.sh
    fi

    if [ "$1" -eq 4 ]; then
	    echo "  Launching ONLY expes 4 : Default of Credit Card Clients dataset, min TPR"
		sbatch batch_default_credit_min_tpr.sh
    fi

fi
