"""
Forwads model and saves its outputs to the specified file
"""

from argparse import ArgumentParser
import torch
import transformers
import data_readers

from vectorizing import get_all_vectors

def main():
    args_parser = ArgumentParser()
    args_parser.add_argument('--model', type=str)
    args_parser.add_argument('--tokenizer', type=str, default=None)
    args_parser.add_argument('--datapath', type=str)
    args_parser.add_argument('--output_file', type=str)

    args_parser.add_argument('--batch_size', type=int, default=64)

    # DataReader arguments
    args_parser.add_argument('--max_length', type=int, default=80)
    args_parser.add_argument('--replace_word_with_mask', type=bool, default=False)

    args = args_parser.parse_args()

    model_name = args.model
    tokenizer_name = args.tokenizer or model_name
    tokenizer_class = transformers.XLMRobertaTokenizerFast if 'xlm' in tokenizer_name.lower() else transformers.BertTokenizerFast

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    print('Loading model')
    model = transformers.AutoModel.from_pretrained(model_name).to(device)
    print('Loading tokenizer')
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    print('Loading data')
    datareader = data_readers.BtsRncReader(args.datapath, tokenizer,
                                           max_length = args.max_length,
                                           replace_word_with_mask = args.replace_word_with_mask)
    dataset = datareader.create_dataset()

    print('Forwarding data')
    all_hidden_states = get_all_vectors(dataset, model, batch_size=args.batch_size)

    print('Number of layers:', len(all_hidden_states))
    print('Shape:', all_hidden_states[0].shape)

    print('Saving data into', args.output_file)
    torch.save(all_hidden_states, args.output_file)



if __name__ == '__main__':
    main()