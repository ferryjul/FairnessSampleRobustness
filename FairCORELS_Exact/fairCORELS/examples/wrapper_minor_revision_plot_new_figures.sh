for METRIC in 1 3 4 5
    do
    echo Computing results for metric $METRIC
    for EPSILON in 0.98 0.985 0.99 0.995
        do # parallel (background) calls

        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=840 --metric=$METRIC --show=0 &

        python3 plot_script_clean_new.py --dataset=adult --epsilon=$EPSILON --plotType=420 --metric=$METRIC --show=0 & 
        pids[0]=$!
        python3 plot_script_clean_new.py --dataset=compas --epsilon=$EPSILON --plotType=420 --metric=$METRIC --show=0 &
        pids[1]=$!
        python3 plot_script_clean_new.py --dataset=default_credit --epsilon=$EPSILON --plotType=420 --metric=$METRIC --show=0 &
        pids[2]=$!
        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=420 --metric=$METRIC --show=0 &
        pids[3]=$!
        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
    done
done