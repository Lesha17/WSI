import pandas
import tqdm
import torch
import re
import os


def get_word_token_ids(all_token_positions, word_positions):
    all_positions_iter = enumerate(all_token_positions)
    i, sos_pos = next(all_positions_iter, (None, None))
    i, current_pos = next(all_positions_iter, (None, None))
    result = []
    for word_start, word_end in word_positions:
        while current_pos is not None and current_pos[0] < word_start:
            i, current_pos = next(all_positions_iter, (None, None))
        while current_pos is not None and 0 < current_pos[1] <= word_end:
            result.append(i)
            i, current_pos = next(all_positions_iter, (None, None))

    return result


class BaseDataReader:
    def __init__(self):
        self.dataframe = None
        self.labels_dataframe = None

    def create_dataset(self, word: str = None):
        raise NotImplementedError()

    def get_words(self):
        raise NotImplementedError()

    def get_dataframe(self):
        if self.dataframe is None:
            self.dataframe = self.create_dataframe()
        return self.dataframe

    def create_dataframe(self):
        raise NotImplementedError()

    def get_labels_dataframe(self):
        if self.labels_dataframe is None:
            self.labels_dataframe = self.create_labels_dataframe()
        return self.labels_dataframe

    def create_labels_dataframe(self):
        raise NotImplementedError()

    def set_predict_labels(self, labels):
        df = self.get_dataframe()
        df['predict_sense_id'] = labels

    def get_word_df_mask(self, word: str):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class BtsRncReader(BaseDataReader):

    def __init__(self, datapath: str, tokenizer,
                 progress: bool = False,
                 max_length: int = 80,
                 replace_word_with_mask: bool = False,
                 several_mask_tokens: bool = False,
                 word_and_pattern: bool = False):
        self.datapath = datapath
        self.tokenizer = tokenizer
        self.progress = progress
        self.max_length = max_length
        self.replace_word_with_mask = replace_word_with_mask
        self.several_mask_tokens = several_mask_tokens
        self.word_and_pattern = word_and_pattern

        self.dataframe = None

    def create_dataset(self, word: str = None):
        df = self.get_dataframe()

        if word is not None:
            df = df[self.get_word_df_mask(word)]

        data = df.iterrows()

        results = []
        if self.progress:
            tqdm.tqdm(data, total=len(df))
        for index, row in data:
            context = row['context'].replace('№', 'N')

            if self.replace_word_with_mask:
                masked_context = ''
                prev_end_pos = 0
                positions_str = re.split(r',\s*', row['positions'])
                for pos_str in positions_str:
                    start_pos, end_pos = (int(p) for p in pos_str.split('-'))
                    masked_context += context[prev_end_pos:start_pos]
                    word = context[start_pos:end_pos + 1]
                    if self.word_and_pattern:
                        masked_context += word + " и "
                    if self.several_mask_tokens:  # TODO handle it correctly
                        masked_context += ' '.join(
                            [self.tokenizer.mask_token] * len(self.tokenizer.encode(word, add_special_tokens=False)))
                    else:
                        masked_context += self.tokenizer.mask_token
                    prev_end_pos = end_pos + 1
                masked_context += context[prev_end_pos:]
                encodings = self.tokenizer.encode_plus(masked_context,
                                                       return_tensors='pt',
                                                       padding='max_length',
                                                       max_length=self.max_length,
                                                       return_offsets_mapping=False)
                encodings['given_word_mask'] = encodings['input_ids'] == self.tokenizer.mask_token_id
            else:
                encodings = self.tokenizer.encode_plus(context,
                                                       return_tensors='pt',
                                                       padding='max_length',
                                                       max_length=self.max_length,
                                                       return_offsets_mapping=True)
                word_indices = get_word_indices(row, encodings)
                given_word_mask = torch.zeros_like(encodings['attention_mask'])
                given_word_mask[0][word_indices] = 1
                encodings['given_word_mask'] = given_word_mask
                del encodings['offset_mapping']

            encodings = {k: v.squeeze() for k, v in encodings.items()}
            encodings['label'] = row['gold_sense_id']
            results.append(encodings)

        return results

    def get_words(self):
        return self.get_dataframe().word.unique()

    def create_dataframe(self):
        return pandas.read_csv(self.datapath, sep='\t')

    def create_labels_dataframe(self):
        return self.get_dataframe()

    def get_word_df_mask(self, word: str):
        df = self.get_dataframe()
        return df.word == word

    def __len__(self):
        return len(self.get_dataframe())


class SemEval2013Reader(BaseDataReader):
    PATTERN = re.compile(r'<b>(.+?)</b>')

    def __init__(self, datapath, tokenizer,
                 max_length: int = 512):
        self.datapath = datapath
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataframe = None
        self.labels_dataframe = None
        self.words_dataframe = None

    @staticmethod
    def get_match_positions(match):
        outer_span = match.span(0)
        inner_span = match.span(1)
        return outer_span[0], inner_span[0], inner_span[1], outer_span[1]

    @staticmethod
    def get_word_spans_positions(context):
        return [SemEval2013Reader.get_match_positions(m) for m in SemEval2013Reader.PATTERN.finditer(context)]

    @staticmethod
    def clear_tags_and_get_positions(context):
        word_spans_positions = SemEval2013Reader.get_word_spans_positions(context)
        prev_position = 0
        parts = []
        positions = []
        current_diff = 0
        for outer_start, inner_start, inner_end, outer_end in word_spans_positions:
            parts.append(context[prev_position:outer_start])
            parts.append(context[inner_start:inner_end])
            prev_position = outer_end

            current_diff += inner_start - outer_start
            positions.append((inner_start - current_diff, inner_end - current_diff))
            current_diff += outer_end - inner_end
        parts.append(context[prev_position:])
        return ''.join(parts), positions

    def create_dataset(self, word: str = None):
        df = self.get_dataframe()
        if word is not None:
            word_df_mask = self.get_word_df_mask(word)
            df = df[word_df_mask]
        result = []
        for index, row in df.iterrows():
            context = row.snippet
            if pandas.isna(context):
                context = ''

            clear_context, word_positions = SemEval2013Reader.clear_tags_and_get_positions(context)

            encodings = self.tokenizer.encode_plus(clear_context,
                                                   return_tensors='pt',
                                                   padding='max_length',
                                                   truncation=True,
                                                   max_length=self.max_length,
                                                   return_offsets_mapping=True)
            encodings = {k: v.squeeze() for k, v in encodings.items()}
            word_token_ids = get_word_token_ids(encodings['offset_mapping'], word_positions)

            encodings['given_word_mask'] = torch.zeros_like(encodings['input_ids'])
            encodings['given_word_mask'][word_token_ids] = 1
            encodings['word_token_ids'] = word_token_ids

            encodings['label'] = df.loc[index].gold_sense_id
            result.append(encodings)

        return result

    def get_word_df_mask(self, word: str):
        df = self.get_dataframe()
        word_df = self._get_words_dataframe()
        word_id = word_df[word_df.description == word].index[0]
        return df.word_id == word_id

    def create_dataframe(self):
        datapath = os.path.join(self.datapath, 'results.txt')
        dataframe = pandas.read_csv(datapath, sep='\t', dtype={'ID': str})
        dataframe = dataframe.dropna(subset=['snippet'])
        dataframe['word_id'] = dataframe['ID'].apply(lambda id: int(id.split('.')[0]))
        dataframe['snippet_id'] = dataframe['ID'].apply(lambda id: int(id.split('.')[1]))
        dataframe = dataframe.set_index('ID')
        labels_df = self.get_labels_dataframe()
        if 'gold_sense_id' in labels_df:
            dataframe['gold_sense_id'] = labels_df.loc[dataframe.index, 'gold_sense_id']
        return dataframe

    def create_labels_dataframe(self):
        datapath = os.path.join(self.datapath, 'STRel.txt')
        labels_dataframe = pandas.read_csv(datapath, sep='\t', dtype={'subTopicID': str, 'resultID': str})
        labels_dataframe['word_id'] = labels_dataframe['subTopicID'].apply(
            lambda id: int(id.split('.')[0]))
        labels_dataframe['gold_sense_id'] = labels_dataframe['subTopicID'].apply(
            lambda id: int(id.split('.')[1]))
        labels_dataframe = labels_dataframe.set_index('resultID')
        return labels_dataframe

    def _get_words_dataframe(self):
        if self.words_dataframe is None:
            datapath = os.path.join(self.datapath, 'topics.txt')
            self.words_dataframe = pandas.read_csv(datapath, sep='\t', index_col='id')
        return self.words_dataframe

    def get_words(self):
        word_df = self._get_words_dataframe()
        return word_df.description.unique()

    def __len__(self):
        return len(self.get_dataframe())


if __name__ == '__main__':
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    datapath = 'data/semeval-2013_task11_dataset'
    datareader = SemEval2013Reader(datapath, tokenizer, replace_word_with_mask=True)
    dataset = datareader.create_dataset()
    print(len(dataset))
    print(dataset[0])

    df = datareader.get_dataframe()
    df[datareader.get_word_df_mask('polaroid'), 'xyu'] = 'pidor'
    print(df['xyu'])
