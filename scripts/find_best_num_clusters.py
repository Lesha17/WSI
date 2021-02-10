import torch
import transformers
from argparse import ArgumentParser

import data_readers
from clustering import make_labeling
from metrics import calculate_metrics_on_labeled_data


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--tokenizer', type=str)
    arg_parser.add_argument('--bert_out_file', type=str)
    arg_parser.add_argument('--dataset_type', type=str, default='bts-rnc')
    arg_parser.add_argument('--datapath', type=str)

    arg_parser.add_argument('--max_length', type=int, default=128)
    arg_parser.add_argument('--replace_word_with_mask', type=bool, default=True)
    arg_parser.add_argument('--bert_layer', type=int, default=-1)
    arg_parser.add_argument('--num_clusters_min', type=int, default=2)
    arg_parser.add_argument('--num_clusters_max', type=int, default=30)
    arg_parser.add_argument('--target_metric', type=str, default='ARI')

    arg_parser.add_argument('--result_message_file', type=str, default=None)

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
        datareader = data_readers.SemEval2013Reader(args.datapath, tokenizer,
                                                    max_length=args.max_length,
                                                    replace_word_with_mask=args.replace_word_with_mask)
    else:
        raise AttributeError('Unsupported dataset type: ' + args.dataset_type)


    target_metric = args.target_metric
    best_metric_value = 0
    best_cluster_num = 0
    for num_clusters in range(args.num_clusters_min, args.num_clusters_max + 1):
        print(f'Labelling data with {num_clusters} clusters')
        labels = make_labeling(datareader, bert_out, bert_layer=args.bert_layer, num_clusters=num_clusters)
        datareader.set_predict_labels(labels)
        metrics = calculate_metrics_on_labeled_data(datareader, datareader)
        metric_value = metrics[target_metric]

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_cluster_num = num_clusters

    message = f'Best {target_metric}: {best_metric_value} with {best_cluster_num} clusters'
    print(message)
    if args.result_message_file is not None:
        with open(args.result_message_file, 'w') as f:
            f.write(message)


if __name__ == '__main__':
    main()
