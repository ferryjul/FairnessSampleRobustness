#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Merging results for all 4 datasets"
    for DATASET in 1 2 3 4
    do
        if [ "$DATASET" -eq 1 ]; then
            echo "  Merging results for the adult dataset"
        fi

        if [ "$DATASET" -eq 2 ]; then
            echo "  Merging results for the COMPAS dataset"
        fi

        if [ "$DATASET" -eq 3 ]; then
            echo "  Merging results for the Marketing dataset"
        fi

        if [ "$DATASET" -eq 4 ]; then
            echo "  Merging results for the Default Credit dataset"
        fi
        python3 gen_csv_paretos_complete.py --metric=1 --dataset=$DATASET --filtering=0
        python3 gen_csv_paretos_complete.py --metric=2 --dataset=$DATASET --filtering=0
        python3 gen_csv_paretos_complete.py --metric=3 --dataset=$DATASET --filtering=0
        python3 gen_csv_paretos_complete.py --metric=4 --dataset=$DATASET --filtering=0
        python3 gen_csv_paretos_complete.py --metric=5 --dataset=$DATASET --filtering=0
        python3 gen_csv_paretos_complete.py --metric=6 --dataset=$DATASET --filtering=0
    done

fi

if [ "$#" -eq 1 ]; then
    if [ "$1" -eq 1 ]; then
        echo "Merging results for the adult dataset only"
    fi

    if [ "$1" -eq 2 ]; then
        echo "Merging results for the COMPAS dataset only"
    fi

    if [ "$1" -eq 3 ]; then
        echo "Merging results for the Marketing dataset only"
    fi

    if [ "$1" -eq 4 ]; then
        echo "Merging results for the Default Credit dataset only"
    fi
    python3 gen_csv_paretos_complete.py --metric=1 --dataset=$1 --filtering=0
    python3 gen_csv_paretos_complete.py --metric=2 --dataset=$1 --filtering=0
    python3 gen_csv_paretos_complete.py --metric=3 --dataset=$1 --filtering=0
    python3 gen_csv_paretos_complete.py --metric=4 --dataset=$1 --filtering=0
    python3 gen_csv_paretos_complete.py --metric=5 --dataset=$1 --filtering=0
    python3 gen_csv_paretos_complete.py --metric=6 --dataset=$1 --filtering=0
fi
