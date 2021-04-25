#!/usr/bin/env bash

for tokenizer in bert-base-cased  bert-base-uncased # bert-large-cased  bert-large-uncased
do
  for masking_type in all_word_tokens dont_mask first_token next_token previous_and_next_token previous_token word_and_mask mask_and_word
  do
    for word_vector_fn in avg_word_tokens_vector avg_context_vector avg_context_without_word_tokens first_word_token_vector first_context_vector previous_token_vector next_token_vector
    do
      for layer_num in {1..12}
      do
        run_file=$1/$tokenizer/$masking_type/$word_vector_fn/layer_$layer_num
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
                  --bert_layer=$layer_num \
                  --result_json_file=$result_json_file
        fi
      done
    done
  done
done