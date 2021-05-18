import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
import transformers
import pandas as pd
from scipy.spatial.distance import cdist

import data_readers
import vectorizing
from metrics import calculate_metrics, calculate_metrics_on_labeled_data


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--tokenizer', type=str)
    arg_parser.add_argument('--bert_out_file', type=str)
    arg_parser.add_argument('--subtopic_embeds_bert_out_file', type=str)
    arg_parser.add_argument('--dataset_type', type=str, default='bts-rnc')
    arg_parser.add_argument('--word_vector_fn', type=str, default='avg_word_tokens_vector',
                            choices=vectorizing.WORD_VECTOR_FNS.keys())
    arg_parser.add_argument('--subtopics_word_vector_fn', type=str, default='avg_word_tokens_vector',
                            choices=vectorizing.WORD_VECTOR_FNS.keys())
    arg_parser.add_argument('--datapath', type=str)

    arg_parser.add_argument('--max_length', type=int, default=128)
    arg_parser.add_argument('--bert_layer', type=int, default=-1)
    arg_parser.add_argument('--target_metric', type=str, default='ARI')

    arg_parser.add_argument('--result_message_file', type=str, default=None)
    arg_parser.add_argument('--result_json_file', type=str, default=None)

    args = arg_parser.parse_args()
    tokenizer_name = args.tokenizer
    tokenizer_class = transformers.XLMRobertaTokenizerFast if 'xlm' in tokenizer_name.lower() else transformers.BertTokenizerFast
    print('Loading tokenizer')
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name + "_")

    print('Loading bert out')
    bert_out = torch.load(args.bert_out_file)
    subtopics_embeds = torch.load(args.subtopic_embeds_bert_out_file)

    print('Loading data')
    datareader = data_readers.SemEval2013Reader(args.datapath, tokenizer, max_length=args.max_length)
    subtopics_dr = data_readers.SemEval2013SubTopicsReader(args.datapath, tokenizer,
                                                           max_length=subtopics_embeds[-1].shape[1])

    df = datareader.get_dataframe()
    subtopics_df = subtopics_dr.get_dataframe()

    result = pd.Series([-1] * len(datareader), index=df.index)

    dataset = datareader.create_dataset()
    subtopic_dataset = subtopics_dr.create_dataset()

    contexts_word_vector_fn = vectorizing.WORD_VECTOR_FNS[args.word_vector_fn]
    subtopics_word_vector_fn = vectorizing.WORD_VECTOR_FNS[args.subtopics_word_vector_fn]

    word_vectors = contexts_word_vector_fn(bert_out, dataset, bert_layer=args.bert_layer)
    subtopic_vectors = subtopics_word_vector_fn(subtopics_embeds, subtopic_dataset, bert_layer=args.bert_layer)

    word_vector_norms = torch.norm(word_vectors, dim=-1)
    nonzero_words = word_vector_norms > 0

    for word_id in df.word_id.unique():
        word_df_mask = (df.word_id == word_id) & nonzero_words
        subtopic_df_mask = (subtopics_df.word_id == word_id)

        current_word_vectors = word_vectors[word_df_mask]
        current_subtopic_vectors = subtopic_vectors[subtopic_df_mask]

        distances = cdist(current_word_vectors, current_subtopic_vectors, metric='cosine')
        best_match = np.argmin(distances, axis=1)
        labels = subtopics_df[subtopic_df_mask].gold_sense_id[best_match]
        result[word_df_mask] = labels

    datareader.set_predict_labels(result)
    metrics = calculate_metrics_on_labeled_data(datareader, datareader)

    print(f'Best {args.target_metric}: {metrics[args.target_metric]}')

    if args.result_json_file is not None:
        os.makedirs(os.path.dirname(args.result_json_file), exist_ok=True)
        with open(args.result_json_file, 'w') as f:
            result_json = {'bert_out_file': args.bert_out_file, 'bert_layer': args.bert_layer,
                           'target_metric': args.target_metric,
                           'metric_value': metrics[args.target_metric]}
            json.dump(result_json, f)


if __name__ == '__main__':
    main()