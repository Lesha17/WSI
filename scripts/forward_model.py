"""
Forwads model and saves its outputs to the specified file
"""

from argparse import ArgumentParser
import torch
import transformers

import context_masking
import data_readers

import vectorizing

def main():
    args_parser = ArgumentParser()
    args_parser.add_argument('--model', type=str)
    args_parser.add_argument('--tokenizer', type=str, default=None)
    args_parser.add_argument('--result_type', type=str, default=vectorizing.RESULT_TYPE_ALL_HIDDEN,
                             choices=[vectorizing.RESULT_TYPE_POOLER_OUT, vectorizing.RESULT_TYPE_ALL_HIDDEN])
    args_parser.add_argument('--context_masker', type=str, default='dont_mask')
    args_parser.add_argument('--dataset_type', type=str, default='bts-rnc')
    args_parser.add_argument('--datapath', type=str)
    args_parser.add_argument('--output_file', type=str)

    args_parser.add_argument('--batch_size', type=int, default=64)

    # DataReader arguments
    args_parser.add_argument('--max_length', type=int, default=None)

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

    context_masker = context_masking.CONTEXT_MASKERS[args.context_masker]
    max_len = args.max_length or model.config.max_position_embeddings

    print('Loading data')
    if args.dataset_type == 'bts-rnc':
        datareader = data_readers.BtsRncReader(args.datapath, tokenizer,
                                           max_length = max_len)
    elif args.dataset_type == 'semeval-2013':
        datareader = data_readers.SemEval2013Reader(args.datapath, tokenizer,
                                               max_length=max_len)
    elif args.dataset_type == 'semeval-2013-subtopics':
        datareader = data_readers.SemEval2013SubTopicsReader(args.datapath, tokenizer, max_length=max_len)
    else:
        raise AttributeError('Unsupported dataset type: ' + args.dataset_type)
    dataset = datareader.create_dataset()
    dataset = context_masking.apply_context_masking_to_dataset(dataset, context_masker, tokenizer)

    print('Forwarding data')
    all_hidden_states = vectorizing.get_all_vectors(dataset, model, batch_size=args.batch_size,
                                                    result_type=args.result_type)

    print('Number of layers:', len(all_hidden_states))
    print('Shape:', all_hidden_states[0].shape)

    print('Saving data into', args.output_file)
    torch.save(all_hidden_states, args.output_file)



if __name__ == '__main__':
    main()