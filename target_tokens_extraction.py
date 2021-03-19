import torch


def masked_tokens(model_inputs, mask_token_id, word_mask):
    return model_inputs['input_ids'] == mask_token_id


def word_tokens(model_inputs, mask_token_id, word_mask):
    return word_mask


def all_tokens(model_inputs, mask_token_id, word_mask):
    return model_inputs['attention_mask']


def all_unmasked_tokens(model_inputs, mask_token_id, word_mask):
    return model_inputs['attention_mask'] * (model_inputs['input_ids'] != mask_token_id)


def all_tokens_except_word(model_inputs, mask_token_id, word_mask):
    return model_inputs['attention_mask'] * (word_mask == 0)
