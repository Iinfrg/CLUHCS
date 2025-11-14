#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
PYTHON_CMD="python"  # 确保你的 python 命令指向 Python 3.x
SCRIPT_PATH="main.py"
log_file="run_log.txt"

> $log_file

for dataset in acm
do
    for u in 0.8
    do
        for t_hops in 6
        do
            for t_n_layers in 1 2 3 4 5 6 7
            do
                echo "Running: dataset=$dataset, u=$u, t_hops=$t_hops, t_n_layers=$t_n_layers"
                $PYTHON_CMD $SCRIPT_PATH $dataset \
                    --gpu 0 \
                    --u $u \
                    --t_hops $t_hops \
                    --t_n_layers $t_n_layers 2>&1 | tee -a $log_file
            done
        done
    done
done