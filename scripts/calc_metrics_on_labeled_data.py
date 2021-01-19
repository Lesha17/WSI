import pandas
from argparse import ArgumentParser
from metrics import calculate_metrics_on_labeled_data

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--gold_path')
    arg_parser.add_argument('--labeled_path')

    args = arg_parser.parse_args()

    gold_df = pandas.read_csv(args.gold_path, sep='\t')
    labeled_df = pandas.read_csv(args.labeled_path, sep='\t')

    metrics = calculate_metrics_on_labeled_data(gold_df, labeled_df)
    print(metrics)