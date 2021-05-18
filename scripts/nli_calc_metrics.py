import json
import os
from collections import defaultdict

import numpy
import pandas
import transformers
import torch
from argparse import ArgumentParser

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

import data_readers

TITLE = 'title'
SNIPPET = 'snippet'
SUBTOPIC_DESCRIPTION = 'subtopic_description'

MODEL_2_CLS = {
    'bert-base-uncased': transformers.AutoModelForNextSentencePrediction,
    'bert-base-cased': transformers.AutoModelForNextSentencePrediction,
    'roberta-large-mnli': transformers.AutoModelForSequenceClassification,
    'facebook/bart-large-mnli': transformers.AutoModelForSequenceClassification
}


def find_batch_size(num_contexts, num_subtopics):
    result = num_contexts
    while result * num_subtopics > max(64, num_subtopics):
        result -= 1
    return result


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--tokenizer', type=str)
    arg_parser.add_argument('--dataset_type', type=str, default='semeval-2013')
    arg_parser.add_argument('--datapath', type=str)
    arg_parser.add_argument('--target_logit_id', type=int, default=0)
    arg_parser.add_argument('--score_as_dist', type=bool, default=False)

    arg_parser.add_argument('--left_context', type=str, choices=[TITLE, SNIPPET, SUBTOPIC_DESCRIPTION], default=SNIPPET)
    arg_parser.add_argument('--right_context', type=str, choices=[TITLE, SNIPPET, SUBTOPIC_DESCRIPTION],
                            default=SNIPPET)

    arg_parser.add_argument('--max_length', type=int, default=512)
    arg_parser.add_argument('--target_metric', type=str, default='ARI')

    arg_parser.add_argument('--num_clusters_min', type=int, default=2)
    arg_parser.add_argument('--num_clusters_max', type=int, default=30)

    arg_parser.add_argument('--result_json_file', type=str, default=None)
    arg_parser.add_argument('--result_csv', type=str, default=None)

    args = arg_parser.parse_args()

    print(args)

    tokenizer_name = args.tokenizer

    print(f'Preloading tokenizer {tokenizer_name}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name + '_')
    print(f'Preloading model {tokenizer_name}')
    model_cls = MODEL_2_CLS[tokenizer_name]
    model = model_cls.from_pretrained(tokenizer_name)
    if torch.cuda.is_available():
        model = model.cuda()

    device = 0 if torch.cuda.is_available() else -1

    datareader = data_readers.SemEval2013Reader(args.datapath, tokenizer, max_length=args.max_length)
    topics_datareader = data_readers.SemEval2013SubTopicsReader(args.datapath, tokenizer, max_length=args.max_length)
    df = datareader.create_dataframe()
    topics_df = topics_datareader.create_dataframe()
    result = pandas.Series([-1] * len(datareader), index=df.index)

    with_clustering = args.left_context in [TITLE, SNIPPET] and args.right_context in [TITLE, SNIPPET]

    if with_clustering:
        softmax_ari_per_num_clusters = defaultdict(list)
        logits_ari_per_num_clusters = defaultdict(list)
        for word_id in df.word_id.unique():
            print(f'Processinf word #{word_id}')
            word_df_mask = (df.word_id == word_id)

            snippets = df[word_df_mask].snippet
            titles = df[word_df_mask].title

            if args.left_context == TITLE:
                left_contexts = titles
            elif args.left_context == SNIPPET:
                left_contexts = snippets
            else:
                raise AttributeError('Unsupported left context for clustering:', args.left_context)

            if args.right_context == TITLE:
                right_contexts = titles
            elif args.right_context == SNIPPET:
                right_contexts = snippets
            else:
                raise AttributeError('Unsupported right context for clustering:', args.right_context)

            softmax_distances_matrix = numpy.zeros(shape=(len(snippets), len(snippets)))
            logits_matrix = numpy.zeros(shape=(len(snippets), len(snippets)))

            bs = 64
            for i, ctx in enumerate(left_contexts):
                ctx_logits = []
                for j in range((len(right_contexts) - 1) // bs + 1):
                    batch_contexts = right_contexts[j * bs:(j + 1) * bs]
                    # batch_results = mothefucking_pipeline(ctx, batch_contexts, hypothesis_template=' the same as {}')

                    enc = tokenizer.batch_encode_plus(list(zip([ctx] * len(batch_contexts), batch_contexts)),
                                                      return_tensors='pt',
                                                      padding=True, truncation=True, max_length=args.max_length)
                    enc = {k: v.to(model.device) for k, v in enc.items()}
                    with torch.no_grad():
                        model_outs = model(**enc)
                    batch_logits = model_outs['logits'][:, args.target_logit_id].cpu().detach()

                    ctx_logits += batch_logits

                ctx_scores = numpy.exp(ctx_logits) / numpy.exp(ctx_logits).sum(-1, keepdims=True)

                for ctx_idx, score_logit in enumerate(zip(ctx_scores, ctx_logits)):
                    score, logit = score_logit
                    if args.score_as_dist:
                        softmax_distances_matrix[i, ctx_idx] = score
                    else:
                        softmax_distances_matrix[i, ctx_idx] = 1 - score
                    logits_matrix[i, ctx_idx] = logit

                softmax_distances_matrix[i, i] = 0
                logits_matrix[i, i] = 0

            if args.score_as_dist:
                logits_dist_matrix = logits_matrix + numpy.min(logits_matrix) * numpy.identity(logits_matrix.shape[0]) \
                                     - numpy.min(logits_matrix)
            else:
                logits_dist_matrix = numpy.max(logits_matrix) \
                                     - (logits_matrix + numpy.max(logits_matrix) * numpy.identity(logits_matrix.shape[0]))

            for num_clusters in range(args.num_clusters_min, args.num_clusters_max + 1):
                clusterer = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
                try:
                    labels = clusterer.fit_predict(softmax_distances_matrix)
                    gold_labels = df[word_df_mask].gold_sense_id
                    softmax_ari_score = adjusted_rand_score(gold_labels, labels)
                    softmax_ari_per_num_clusters[num_clusters].append(softmax_ari_score)
                except Exception as e:
                    print(f'Cannot cluster softmax with {num_clusters} clusters for {word_id}')

                try:
                    clusterer = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
                    labels = clusterer.fit_predict(logits_dist_matrix)
                    gold_labels = df[word_df_mask].gold_sense_id
                    logits_ari_score = adjusted_rand_score(gold_labels, labels)
                    logits_ari_per_num_clusters[num_clusters].append(logits_ari_score)
                except Exception as e:
                    print(f'Cannot cluster logits with {num_clusters} clusters for {word_id}')

            print(word_id, softmax_ari_per_num_clusters)
            print(word_id, logits_ari_per_num_clusters)

        softmax_aris = [0] * len(softmax_ari_per_num_clusters)
        for nc, scores in softmax_ari_per_num_clusters.items():
            softmax_aris[nc - args.num_clusters_min] = numpy.mean(scores)
        logits_aris = [0] * len(logits_ari_per_num_clusters)
        for nc, scores in logits_ari_per_num_clusters.items():
            logits_aris[nc - args.num_clusters_min] = numpy.mean(scores)

        best_softmax_nc = numpy.argmax(softmax_aris) + args.num_clusters_min
        best_softmax_ari = softmax_aris[best_softmax_nc - args.num_clusters_min]

        best_logits_nc = numpy.argmax(logits_aris) + args.num_clusters_min
        best_logits_ari = logits_aris[best_logits_nc - args.num_clusters_min]

        if args.result_json_file is not None:
            os.makedirs(os.path.dirname(args.result_json_file), exist_ok=True)
            with open(args.result_json_file, 'w') as f:
                result_json = {
                    'model': tokenizer_name, 'target_logit_id': args.target_logit_id,
                    'left_context': args.left_context, 'right_context': args.right_context,
                    'best_softmax_ari': best_softmax_ari, 'best_softmax_nc': best_softmax_nc,
                    'best_logits_ari': best_logits_ari, 'best_logits_nc': best_logits_nc,
                    'softmax_ari_per_num_clusters': softmax_ari_per_num_clusters,
                    'logits_ari_per_num_clusters': logits_ari_per_num_clusters
                }
                json.dump(result_json, f)

    else:
        aris = []
        for word_id in df.word_id.unique():
            labels = []
            print(f'Processinf word #{word_id}')
            word_df_mask = (df.word_id == word_id)
            subtopic_df_mask = (topics_df.word_id == word_id)

            snipets = df[word_df_mask].snippet
            titles = df[word_df_mask].title
            word_topics_df = topics_df[subtopic_df_mask]
            subtopics = word_topics_df.description

            if args.left_context == TITLE:
                left_contexts = titles
            elif args.left_context == SNIPPET:
                left_contexts = snipets
            else:
                left_contexts = subtopics

            if args.right_context == TITLE:
                right_contexts = titles
            elif args.right_context == SNIPPET:
                right_contexts = snipets
            else:
                right_contexts = subtopics

            bs = 64
            for i, ctx in enumerate(left_contexts):
                ctx_logits = []
                for j in range((len(right_contexts) - 1) // bs + 1):
                    batch_topics = right_contexts[j * bs:(j + 1) * bs]

                    enc = tokenizer.batch_encode_plus(list(zip([ctx] * len(batch_topics), batch_topics)),
                                                      return_tensors='pt',
                                                      padding=True, truncation=True, max_length=512)
                    enc = {k: v.to(model.device) for k, v in enc.items()}
                    with torch.no_grad():
                        model_outs = model(**enc)
                    batch_logits = model_outs['logits'][:, args.target_logit_id].cpu().detach()

                    ctx_logits += batch_logits

                ctx_scores = numpy.exp(ctx_logits) / numpy.exp(ctx_logits).sum(-1, keepdims=True)
                ctx_label = numpy.argmax(ctx_scores)
                labels.append(ctx_label)

            gold_labels = df[word_df_mask].gold_sense_id
            ari = adjusted_rand_score(gold_labels, labels)
            aris.append(ari)
        metric_value = numpy.mean(aris)
        print('ARI', metric_value)

        if args.result_json_file is not None:
            os.makedirs(os.path.dirname(args.result_json_file), exist_ok=True)
            with open(args.result_json_file, 'w') as f:
                result_json = {
                    'model': tokenizer_name, 'target_logit_id': args.target_logit_id,
                    'left_context': args.left_context, 'right_context': args.right_context,
                    'best_ari': metric_value,
                    'ari_per_word': aris
                }
                json.dump(result_json, f)


if __name__ == '__main__':
    main()
