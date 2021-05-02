#!/usr/bin/env bash

for tokenizer in deepset/sentence_bert # bert-large-cased  bert-large-uncased
do
  for masking_type in dont_mask
  do
    for word_vector_fn in sent_emb
      do
        run_file=$1/$tokenizer/$masking_type/$word_vector_fn
        echo "Run file: $run_file"
        result_json_file=clustering_results/$run_file.json
        if [ -f $result_json_file ]
        then
          echo "File $result_json_file already exists"
        else
          log_dir=logs/$run_file
          mkdir -p $log_dir
          sbatch -t 60 -o $log_dir/stdout.log -e $log_dir/stderr.log \
              hpc_scripts/anypython.sh scripts/find_best_num_clusters.py \
                  --dataset_type=$1 --datapath=data/$1 --tokenizer=$tokenizer \
                  --bert_out_file=bert_out/$1/$tokenizer/$masking_type.pt \
                  --word_vector_fn=$word_vector_fn \
                  --result_json_file=$result_json_file
        fi
    done
  done
done