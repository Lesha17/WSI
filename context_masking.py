import torch


def apply_context_masking_to_dataset(dataset, context_masker, mask_token_id):
    for sample in dataset:
        sample['input_ids'] = context_masker(sample['input_ids'], sample['word_token_ids'], mask_token_id)
    return dataset


def mask_first_token(context_tokens, word_tokens_idx, mask_token_id):
    if len(word_tokens_idx) == 0:
        return context_tokens
    context_tokens[word_tokens_idx[0]] = mask_token_id
    return context_tokens


def mask_all_word_tokens(context_tokens, word_tokens_idx, mask_token_id):
    context_tokens[word_tokens_idx] = mask_token_id
    return context_tokens


def mask_previous_token(context_tokens, word_tokens_idx, mask_token_id):
    if len(word_tokens_idx) == 0 or word_tokens_idx[0] < 2:
        return context_tokens
    context_tokens[word_tokens_idx[0] - 1] = mask_token_id
    return context_tokens


def mask_next_token(context_tokens, word_tokens_idx, mask_token_id):
    if len(word_tokens_idx) == 0:
        return context_tokens
    context_tokens[word_tokens_idx[-1] + 1] = mask_token_id
    return context_tokens


def mask_previous_and_next_token(context_tokens, word_tokens_idx, mask_token_id):
    context_tokens = mask_previous_token(context_tokens, word_tokens_idx, mask_token_id)
    context_tokens = mask_next_token(context_tokens, word_tokens_idx, mask_token_id)
    return context_tokens


def mask_each_token_at_once(context_tokens: torch.Tensor, word_tokens_idx, mask_token_id):
    for token_id in word_tokens_idx:
        result = context_tokens.clone()  # maybe there should be .detach(), idk
        result[token_id] = mask_token_id
        yield result


def dont_mask(context_tokens, word_tokens_idx, mask_token_id):
    return context_tokens


CONTEXT_MASKERS = {
    'dont_mask': dont_mask,
    'first_token': mask_first_token,
    'all_word_tokens': mask_all_word_tokens,
    'previous_token': mask_previous_token,
    'next_token': mask_next_token,
    'previous_and_next_token': mask_previous_and_next_token
}
