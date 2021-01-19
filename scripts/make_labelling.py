import torch
import transformers
from argparse import ArgumentParser
from data_readers import BtsRncReader
from clustering import make_labeling


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--tokenizer', type=str)
    arg_parser.add_argument('--bert_out_file', type=str)
    arg_parser.add_argument('--data_file', type=str)
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

    datareader = BtsRncReader(args.data_file, tokenizer, max_length=args.max_length,
                              replace_word_with_mask=args.replace_word_with_mask)

    print('Labelling data')
    labels = make_labeling(datareader, bert_out, bert_layer=args.bert_layer, num_clusters=args.num_clusters)
    df = datareader._get_dataframe()
    df['predict_sense_id'] = labels

    df.to_csv(args.output_file, sep='\t', index=False)


if __name__ == '__main__':
    main()
