import json
import os

import pandas
import transformers
import torch
from argparse import ArgumentParser
import data_readers
from metrics import calculate_metrics_on_labeled_data

def find_batch_size(num_contexts, num_subtopics):
    result = num_contexts
    while result * num_subtopics > max(128, num_subtopics):
        result -= 1
    return result


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--tokenizer', type=str)
    arg_parser.add_argument('--preload_model', type=bool, default=False)
    arg_parser.add_argument('--dataset_type', type=str, default='semeval-2013')
    arg_parser.add_argument('--datapath', type=str)

    arg_parser.add_argument('--max_length', type=int, default=128)
    arg_parser.add_argument('--target_metric', type=str, default='ARI')

    arg_parser.add_argument('--result_message_file', type=str, default=None)
    arg_parser.add_argument('--result_json_file', type=str, default=None)
    arg_parser.add_argument('--result_csv', type=str, default=None)

    args = arg_parser.parse_args()
    tokenizer_name = args.tokenizer

    model = tokenizer_name
    if args.preload_model:
        print(f'Preloading model {tokenizer_name}')
        model = transformers.AutoModel.from_pretrained(tokenizer_name)
        if torch.cuda.is_available():
            model = model.cuda()

    device = 0 if torch.cuda.is_available() else -1
    pipeline = transformers.pipeline('zero-shot-classification', model=model, tokenizer=tokenizer_name, device=device)

    tokenizer = None
    datareader = data_readers.SemEval2013Reader(args.datapath, tokenizer, max_length=args.max_length)
    topics_datareader = data_readers.SemEval2013SubTopicsReader(args.datapath, tokenizer, max_length=args.max_length)

    df = datareader.create_dataframe()
    topics_df = topics_datareader.create_dataframe()
    result = pandas.Series([-1] * len(datareader), index=df.index)

    for word_id in df.word_id.unique():
        word_df_mask = (df.word_id == word_id)
        subtopic_df_mask = (topics_df.word_id == word_id)

        contexts = df[word_df_mask].snippet
        word_topics_df = topics_df[subtopic_df_mask]
        subtopics = topics_df[subtopic_df_mask].description
        subtopic2id = {row.description: row.gold_sense_id for i, row in word_topics_df.iterrows()}

        bs = find_batch_size(len(contexts), len(subtopics))

        labels = []
        with torch.no_grad():
            for i in range((len(contexts) - 1) // bs + 1):
                batch_contexts = contexts[i * bs:(i+1) * bs]
                batch_results = pipeline(batch_contexts, subtopics)
                try:
                    batch_labels = [r['labels'][0] for r in batch_results] if len(batch_contexts) > 1 \
                        else [batch_results['labels'][0]]
                except TypeError as e:
                    print(word_id, bs, i, len(contexts), len(subtopics))
                    print(batch_results)
                    raise e
                labels += batch_labels

        result[word_df_mask] = [subtopic2id[t] for t in labels]

    datareader.set_predict_labels(result)

    if args.result_csv is not None:
        datareader.get_dataframe().to_csv(args.result_csv, sep='\t')

    metrics = calculate_metrics_on_labeled_data(datareader, datareader)
    metric_value = metrics[args.target_metric]

    print(f'All metrics: {metrics}')

    message = f'Best {args.target_metric}: {metric_value}'
    print(message)
    if args.result_message_file is not None:
        with open(args.result_message_file, 'w') as f:
            f.write(message)

    if args.result_json_file is not None:
        os.makedirs(os.path.dirname(args.result_json_file), exist_ok=True)
        with open(args.result_json_file, 'w') as f:
            result_json = {'target_metric': args.target_metric,
                           'metric_value': metric_value}
            json.dump(result_json, f)


if __name__ == '__main__':
    main()