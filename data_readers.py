import pandas
import tqdm
import torch
import re


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


class BtsRncReader:

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
        df = self._get_dataframe()

        if word is not None:
            df = df[self.get_word_df_index(word)]

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
        return self._get_dataframe().word.unique()

    def _get_dataframe(self):
        if self.dataframe is None:
            self.dataframe = pandas.read_csv(self.datapath, sep='\t')
        return self.dataframe

    def get_word_df_index(self, word: str):
        df = self._get_dataframe()
        df_index = df.word == word
        return df_index

    def __len__(self):
        return len(self._get_dataframe())
