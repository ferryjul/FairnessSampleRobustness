#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Merging results for all 3 datasets"
    for DATASET in 1 2 3 4
    do
        if [ "$DATASET" -eq 1 ]; then
            echo "  Merging results for the adult dataset (expe 1)"
	    python3 merge_results_eps0.py --expe=1
        fi

        if [ "$DATASET" -eq 2 ]; then
            echo "  Merging results for the COMPAS dataset (expe 2)"
	    python3 merge_results_eps0.py --expe=2
        fi

        if [ "$DATASET" -eq 3 ]; then
            echo "  Merging results for the Bank Marketing dataset (expe 3)"
	    python3 merge_results_eps0.py --expe=3
        fi

        if [ "$DATASET" -eq 4 ]; then
            echo "Merging results for the Default of Credit Card Clients dataset (expe 4)"
	        python3 merge_results_eps0.py --expe=4
        fi
    done

fi

if [ "$#" -eq 1 ]; then
    if [ "$1" -eq 1 ]; then
        echo "Merging results for the adult dataset (expe 1) only"
	    python3 merge_results_eps0.py --expe=1
    fi

    if [ "$1" -eq 2 ]; then
        echo "Merging results for the COMPAS dataset (expe 2) only"
	    python3 merge_results_eps0.py --expe=2
    fi

    if [ "$1" -eq 3 ]; then
        echo "  Merging results for the Bank Marketing dataset (expe 3)"
	    python3 merge_results_eps0.py --expe=3
    fi

    if [ "$1" -eq 4 ]; then
        echo "Merging results for the Default of Credit Card Clients dataset (expe 4)"
	    python3 merge_results_eps0.py --expe=4
    fi
fi
