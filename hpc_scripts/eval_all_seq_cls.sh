#!/usr/bin/env bash

for tokenizer in bert-base-cased bert-base-uncased bert-large-cased bert-large-uncased roberta-large-mnli facebook/bart-large-mnli; do
  logit_ids=(0 1)
  if [[ "$tokenizer" == "roberta-large-mnli" ]] || [[ "$tokenizer" == "facebook/bart-large-mnli" ]]; then
    logit_ids=(0 1 2)
  fi
  for logit_id in ${logit_ids[*]}; do
    for left_ctx in title snippet subtopic_description; do
      for right_ctx in title snippet subtopic_description; do
        for score_as_dist in False True; do
          run_file=$1/$tokenizer/logit_"$logit_id"_$score_as_dist/"$left_ctx"_"$right_ctx"
          echo "Run file: $run_file"
          result_json_file=clustering_results/$run_file.json
          if [ -f $result_json_file ]; then
            echo "File $result_json_file already exists"
          else
            log_dir=logs/$run_file
            mkdir -p $log_dir
            sbatch -G 1 -o $log_dir/stdout.log -e $log_dir/stderr.log \
              hpc_scripts/anypython.sh scripts/nli_calc_metrics.py \
              --dataset_type=$1 --datapath=data/$1 --tokenizer=$tokenizer \
              --target_logit_id=$logit_id \
              --score_as_dist=$score_as_dist \
              --left_context=$left_ctx \
              --right_context=$right_ctx \
              --result_json_file=$result_json_file
          fi
        done
      done
    done
  done
done
