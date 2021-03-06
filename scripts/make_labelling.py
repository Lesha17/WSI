import torch
import transformers
from argparse import ArgumentParser
import data_readers
from clustering import make_labeling


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--tokenizer', type=str)
    arg_parser.add_argument('--bert_out_file', type=str)
    arg_parser.add_argument('--dataset_type', type=str, default='bts-rnc')
    arg_parser.add_argument('--datapath', type=str)
    #arg_parser.add_argument('--data_file', type=str)
    arg_parser.add_argument('--output_file', type=str)

    arg_parser.add_argument('--max_length', type=int, default=80)
    arg_parser.add_argument('--replace_word_with_mask', type=bool, default=False)
    arg_parser.add_argument('--bert_layer', type=int, default=-1)
    arg_parser.add_argument('--num_clusters', type=int, default=2)

    args = arg_parser.parse_args()
    tokenizer_name = args.tokenizer
    tokenizer_class = transformers.XLMRobertaTokenizerFast if 'xlm' in tokenizer_name.lower() else transformers.BertTokenizerFast
    print('Loading tokenizer')
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    print('Loading bert out')
    bert_out = torch.load(args.bert_out_file)

    if args.dataset_type == 'bts-rnc':
        datareader = data_readers.BtsRncReader(args.datapath, tokenizer,
                                           max_length = args.max_length,
                                           replace_word_with_mask = args.replace_word_with_mask)
    elif args.dataset_type == 'semeval-2013':
        datareader = data_readers.SemEval2013Reader(args.datapath, tokenizer,
                                               max_length=args.max_length,
                                               replace_word_with_mask=args.replace_word_with_mask)
    else:
        raise AttributeError('Unsupported dataset type: ' + args.dataset_type)

    print('Labelling data')
    labels = make_labeling(datareader, bert_out, bert_layer=args.bert_layer, num_clusters=args.num_clusters)
    datareader.set_predict_labels(labels)
    df = datareader.get_dataframe()

    df.to_csv(args.output_file, sep='\t', index=False)


if __name__ == '__main__':
    main()
