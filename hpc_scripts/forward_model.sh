#!/usr/bin/env bash

sbatch -G 1 hpc_scripts/anypython.sh scripts/forward_model.py \
    --dataset_type=semeval-2013 --datapath=data/semeval-2013 \
    --max_length=128 \
    --model=$1 --context_masker=$2 \
    --output_file=bert_out/semeval-2013/$1/$2.pt