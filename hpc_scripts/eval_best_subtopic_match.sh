#!/usr/bin/env bash

for tokenizer in deepset/sentence_bert bert-base-uncased; do # bert-large-cased  bert-large-uncased
  snippet_fns=(avg_word_tokens_vector avg_context_vector avg_context_without_word_tokens first_word_token_vector first_context_vector previous_token_vector next_token_vector)
  if [[ "$tokenizer" == "deepset/sentence_bert" ]]; then
    snippet_fns=(sent_emb)
  fi
  for snippet_word_vector_fn in ${snippet_fns[*]}; do
    subtopic_fns=(avg_context_vector first_context_vector)
    if [[ "$tokenizer" == "deepset/sentence_bert" ]]; then
      subtopic_fns=(sent_emb)
    fi
    for subtopic_word_vector_fn in ${subtopic_fns[*]}; do
      layer_nums_range=(1 2 3 4 5 6 7 8 9 10 11 12)
      if [[ "$tokenizer" == "deepset/sentence_bert" ]]; then
        layer_nums_range=(0)
      fi
      for layer_num in ${layer_nums_range[*]}; do
        run_file=$1/$tokenizer/"$snippet_word_vector_fn"_"$subtopic_word_vector_fn"/layer_$layer_num
        echo "Run file: $run_file"
        result_json_file=clustering_results/$run_file.json
        if [ -f $result_json_file ]; then
          echo "File $result_json_file already exists"
        else
          log_dir=logs/$run_file
          mkdir -p $log_dir
          sbatch -t 60 -o $log_dir/stdout.log -e $log_dir/stderr.log \
            hpc_scripts/anypython.sh scripts/best_subtopic_match_calc_metrics.py \
            --dataset_type=$1 --datapath=data/$1 --tokenizer=$tokenizer \
            --bert_out_file=bert_out/$1/$tokenizer/dont_mask.pt \
            --subtopic_embeds_bert_out_file=bert_out/"$1"-subtopics/$tokenizer/dont_mask.pt \
            --word_vector_fn="$snippet_word_vector_fn" \
            --subtopics_word_vector_fn="$subtopic_word_vector_fn" \
            --bert_layer="$layer_num" \
            --result_json_file="$result_json_file"
        fi
      done
    done
  done
done
