import torch
import transformers
from argparse import ArgumentParser

from clustering import make_labeling
from data_readers import BtsRncReader
from metrics import calculate_metrics_on_labeled_data


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--tokenizer', type=str)
    arg_parser.add_argument('--bert_out_file', type=str)
    arg_parser.add_argument('--data_file', type=str)

    arg_parser.add_argument('--max_length', type=int, default=80)
    arg_parser.add_argument('--replace_word_with_mask', type=bool, default=False)
    arg_parser.add_argument('--bert_layer', type=int, default=-1)
    arg_parser.add_argument('--num_clusters_min', type=int, default=2)
    arg_parser.add_argument('--num_clusters_max', type=int, default=30)
    arg_parser.add_argument('--target_metric', type=str, default='ARI')

    args = arg_parser.parse_args()
    tokenizer_name = args.tokenizer
    tokenizer_class = transformers.XLMRobertaTokenizerFast if 'xlm' in tokenizer_name.lower() else transformers.BertTokenizerFast
    print('Loading tokenizer')
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    print('Loading bert out')
    bert_out = torch.load(args.bert_out_file)

    datareader = BtsRncReader(args.data_file, tokenizer, max_length=args.max_length,
                              replace_word_with_mask=args.replace_word_with_mask)

    target_metric = args.target_metric
    best_metric_value = 0
    best_cluster_num = 0
    df = datareader._get_dataframe()
    for num_clusters in range(args.num_clusters_min, args.num_clusters_max + 1):
        print(f'Labelling data with {num_clusters} clusters')
        labels = make_labeling(datareader, bert_out, bert_layer=args.bert_layer, num_clusters=num_clusters)
        df['predict_sense_id'] = labels
        metrics = calculate_metrics_on_labeled_data(df, df)
        metric_value = metrics[target_metric]

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_cluster_num = num_clusters

    print(f'Best {target_metric}: {best_metric_value} with {best_cluster_num} clusters')


if __name__ == '__main__':
    main()
