#!/usr/bin/env bash

sbatch hpc_scripts/anypython.sh scripts/find_best_num_clusters.py \
    --dataset_type=$1 --datapath=data/$1 --tokenizer=$2 \
    --bert_out_file=bert_out/$1/$2.pt \
    --result_message_file=result_messages/$1/$2.txt