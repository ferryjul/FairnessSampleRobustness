for METRIC in 1 3 4 5
    do
    echo Computing results for metric $METRIC
    for EPSILON in 0.98 0.985 0.99 0.995
        do # parallel (background) calls

        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=12 --metric=$METRIC --show=0 &

        python3 plot_script_clean_new.py --dataset=adult --epsilon=$EPSILON --plotType=6 --metric=$METRIC --show=0 & 
        pids[0]=$!
        python3 plot_script_clean_new.py --dataset=compas --epsilon=$EPSILON --plotType=6 --metric=$METRIC --show=0 &
        pids[1]=$!
        python3 plot_script_clean_new.py --dataset=default_credit --epsilon=$EPSILON --plotType=6 --metric=$METRIC --show=0 &
        pids[2]=$!
        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=6 --metric=$METRIC --show=0 &
        pids[3]=$!
        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done

        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=14 --metric=$METRIC --show=0 &

        python3 plot_script_clean_new.py --dataset=adult --epsilon=$EPSILON --plotType=7 --metric=$METRIC --show=0 & 
        pids[0]=$!
        python3 plot_script_clean_new.py --dataset=compas --epsilon=$EPSILON --plotType=7 --metric=$METRIC --show=0 &
        pids[1]=$!
        python3 plot_script_clean_new.py --dataset=default_credit --epsilon=$EPSILON --plotType=7 --metric=$METRIC --show=0 &
        pids[2]=$!
        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=7 --metric=$METRIC --show=0 &
        pids[3]=$!
        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done

        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=22 --metric=$METRIC --show=0 &

        python3 plot_script_clean_new.py --dataset=adult --epsilon=$EPSILON --plotType=11 --metric=$METRIC --show=0 & 
        pids[0]=$!
        python3 plot_script_clean_new.py --dataset=compas --epsilon=$EPSILON --plotType=11 --metric=$METRIC --show=0 &
        pids[1]=$!
        python3 plot_script_clean_new.py --dataset=default_credit --epsilon=$EPSILON --plotType=11 --metric=$METRIC --show=0 &
        pids[2]=$!
        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=11 --metric=$METRIC --show=0 &
        pids[3]=$!
        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done

        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=84 --metric=$METRIC --show=0 &

        python3 plot_script_clean_new.py --dataset=adult --epsilon=$EPSILON --plotType=42 --metric=$METRIC --show=0 & 
        pids[0]=$!
        python3 plot_script_clean_new.py --dataset=compas --epsilon=$EPSILON --plotType=42 --metric=$METRIC --show=0 &
        pids[1]=$!
        python3 plot_script_clean_new.py --dataset=default_credit --epsilon=$EPSILON --plotType=42 --metric=$METRIC --show=0 &
        pids[2]=$!
        python3 plot_script_clean_new.py --dataset=marketing --epsilon=$EPSILON --plotType=42 --metric=$METRIC --show=0 &
        pids[3]=$!
        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
    done
done