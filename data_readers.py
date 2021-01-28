import pandas
import tqdm
import torch
import re
import os


def get_word_indices(row, encodings):
    positions_str = re.split(r',\s*', row['positions'])
    offset_mapping_iter = enumerate(list(encodings['offset_mapping'][0]))
    index, current_mapping = next(offset_mapping_iter)
    word_indices = []
    for pos_str in positions_str:
        start_pos, end_pos = (int(p) for p in pos_str.split('-'))
        while current_mapping is not None and current_mapping[0] < start_pos:
            index, current_mapping = next(offset_mapping_iter, (None, None))
        while current_mapping is not None and current_mapping[1] <= end_pos + 1:
            word_indices.append(index)
            index, current_mapping = next(offset_mapping_iter, (None, None))
    return word_indices


class BaseDataReader:
    def create_dataset(self, word: str = None):
        raise NotImplementedError()

    def get_words(self):
        raise NotImplementedError()

    def get_dataframe(self):
        raise NotImplementedError()

    def get_word_df_index(self, word: str):
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
            df = df.loc[self.get_word_df_index(word)]

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

    def get_dataframe(self):
        if self.dataframe is None:
            self.dataframe = pandas.read_csv(self.datapath, sep='\t')
        return self.dataframe

    def get_word_df_index(self, word: str):
        df = self.get_dataframe()
        return df[df.word == word].index

    def __len__(self):
        return len(self.get_dataframe())


class SemEval2013Reader(BaseDataReader):
    def __init__(self, datapath, tokenizer,
                 replace_word_with_mask: bool = False,
                 max_length: int = 128):
        self.datapath = datapath
        self.tokenizer = tokenizer
        self.replace_word_with_mask = replace_word_with_mask
        self.max_length = max_length

        self.dataframe = None
        self.labels_dataframe = None
        self.words_dataframe = None

    def create_dataset(self, word: str = None):
        df = self.get_dataframe()
        labels_df = self._get_labels_dataframe()
        if word is not None:
            word_df_index = self.get_word_df_index(word)
            df = df.loc[word_df_index]
            labels_df = labels_df.loc[word_df_index]
        pattern = re.compile(r'<b>(\w+.?\w*)</b>')
        result = []
        for index, row in df.iterrows():
            context = row.snippet
            if pandas.isna(context):
                continue
            if self.replace_word_with_mask:
                replaced_with_mask = pattern.sub(self.tokenizer.mask_token, context)
                encodings = self.tokenizer.encode_plus(replaced_with_mask,
                                                       return_tensors='pt',
                                                       padding='max_length',
                                                       truncation = True,
                                                       max_length=self.max_length,
                                                       return_offsets_mapping=False)
                encodings['given_word_mask'] = encodings['input_ids'] == self.tokenizer.mask_token_id

                encodings = {k: v.squeeze() for k, v in encodings.items()}
                encodings['label'] = labels_df.loc[index].sense_id
                result.append(encodings)
            else:
                raise NotImplementedError('Building dataset without replacing target with mask is not supported')
        return result

    def get_word_df_index(self, word: str):
        df = self.get_dataframe()
        word_df = self._get_words_dataframe()
        word_id = word_df[word_df.description == word].index[0]
        return df[df.word_id == word_id].index

    def get_dataframe(self):
        if self.dataframe is None:
            datapath = os.path.join(self.datapath, 'results.txt')
            self.dataframe = pandas.read_csv(datapath, sep='\t', dtype={'ID': str})
            self.dataframe['word_id'] = self.dataframe['ID'].apply(lambda id: int(id.split('.')[0]))
            self.dataframe['snippet_id'] = self.dataframe['ID'].apply(lambda id: int(id.split('.')[1]))
            self.dataframe = self.dataframe.set_index('ID')
        return self.dataframe

    def _get_labels_dataframe(self):
        if self.labels_dataframe is None:
            datapath = os.path.join(self.datapath, 'STRel.txt')
            self.labels_dataframe = pandas.read_csv(datapath, sep='\t', dtype={'subTopicID': str, 'resultID': str})
            self.labels_dataframe['word_id'] = self.labels_dataframe['subTopicID'].apply(
                lambda id: int(id.split('.')[0]))
            self.labels_dataframe['sense_id'] = self.labels_dataframe['subTopicID'].apply(
                lambda id: int(id.split('.')[1]))
            self.labels_dataframe = self.labels_dataframe.set_index('resultID')
        return self.labels_dataframe

    def _get_words_dataframe(self):
        if self.words_dataframe is None:
            datapath = os.path.join(self.datapath, 'topics.txt')
            self.words_dataframe = pandas.read_csv(datapath, sep='\t', index_col='id')
        return self.words_dataframe


if __name__ == '__main__':
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    datapath = 'data/semeval-2013_task11_dataset'
    datareader = SemEval2013Reader(datapath, tokenizer, replace_word_with_mask=True)
    dataset = datareader.create_dataset()
    print(len(dataset))
    print(dataset[0])

    df = datareader.get_dataframe()
    df.loc[datareader.get_word_df_index('polaroid'), 'xyu'] = 'pidor'
    print(df['xyu'])
