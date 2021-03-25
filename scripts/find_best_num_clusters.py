import json
import os

import torch
import transformers
from argparse import ArgumentParser

import data_readers
import vectorizing
from clustering import make_labeling
from metrics import calculate_metrics_on_labeled_data


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--tokenizer', type=str)
    arg_parser.add_argument('--bert_out_file', type=str)
    arg_parser.add_argument('--dataset_type', type=str, default='bts-rnc')
    arg_parser.add_argument('--word_vector_fn', type=str, default='avg_word_tokens_vector', choices=vectorizing.WORD_VECTOR_FNS.keys())
    arg_parser.add_argument('--datapath', type=str)

    arg_parser.add_argument('--max_length', type=int, default=128)
    arg_parser.add_argument('--replace_word_with_mask', type=bool, default=True)
    arg_parser.add_argument('--bert_layer', type=int, default=-1)
    arg_parser.add_argument('--num_clusters_min', type=int, default=2)
    arg_parser.add_argument('--num_clusters_max', type=int, default=30)
    arg_parser.add_argument('--target_metric', type=str, default='ARI')

    arg_parser.add_argument('--result_message_file', type=str, default=None)
    arg_parser.add_argument('--result_json_file', type=str, default=None)

    args = arg_parser.parse_args()
    tokenizer_name = args.tokenizer
    tokenizer_class = transformers.XLMRobertaTokenizerFast if 'xlm' in tokenizer_name.lower() else transformers.BertTokenizerFast
    print('Loading tokenizer')
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    print('Loading bert out')
    bert_out = torch.load(args.bert_out_file)

    print('Loading data')
    if args.dataset_type == 'bts-rnc':
        datareader = data_readers.BtsRncReader(args.datapath, tokenizer,
                                               max_length=args.max_length,
                                               replace_word_with_mask=args.replace_word_with_mask)
    elif args.dataset_type == 'semeval-2013':
        datareader = data_readers.SemEval2013Reader(args.datapath, tokenizer, max_length=args.max_length)
    else:
        raise AttributeError('Unsupported dataset type: ' + args.dataset_type)

    target_metric = args.target_metric
    best_metric_value = 0
    best_cluster_num = 0
    metric_value_per_num_clusters = {}
    word_vector_fn = vectorizing.WORD_VECTOR_FNS[args.word_vector_fn]
    for num_clusters in range(args.num_clusters_min, args.num_clusters_max + 1):
        print(f'Labelling data with {num_clusters} clusters')
        labels = make_labeling(datareader, bert_out, bert_layer=args.bert_layer, num_clusters=num_clusters,
                               word_vector_fn=word_vector_fn)
        datareader.set_predict_labels(labels)
        metrics = calculate_metrics_on_labeled_data(datareader, datareader)
        metric_value = metrics[target_metric]
        metric_value_per_num_clusters[num_clusters] = metric_value

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_cluster_num = num_clusters

    message = f'Best {target_metric}: {best_metric_value} with {best_cluster_num} clusters'
    print(message)
    if args.result_message_file is not None:
        with open(args.result_message_file, 'w') as f:
            f.write(message)

    if args.result_json_file is not None:
        os.makedirs(os.path.dirname(args.result_json_file), exist_ok=True)
        with open(args.result_json_file, 'w') as f:
            result_json = {'bert_out_file': args.bert_out_file, 'bert_layer': args.bert_layer,
                           'target_metric': target_metric,
                           'metric_value': best_metric_value, 'best_num_clusters': best_cluster_num,
                           'metric_value_per_num_clusters': metric_value_per_num_clusters}
            json.dump(result_json, f)


if __name__ == '__main__':
    main()
