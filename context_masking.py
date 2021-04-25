import torch


def apply_context_masking_to_dataset(dataset, context_masker, tokenizer):
    for sample in dataset:
        sample['input_ids'] = context_masker(sample['input_ids'], sample['word_token_ids'], tokenizer)
    return dataset


def mask_first_token(context_tokens, word_tokens_idx, tokenizer):
    if len(word_tokens_idx) == 0:
        return context_tokens
    context_tokens[word_tokens_idx[0]] = tokenizer.mask_token_id
    return context_tokens


def mask_all_word_tokens(context_tokens, word_tokens_idx, tokenizer):
    context_tokens[word_tokens_idx] = tokenizer.mask_token_id
    return context_tokens


def mask_previous_token(context_tokens, word_tokens_idx, tokenizer):
    if len(word_tokens_idx) == 0 or word_tokens_idx[0] < 2:
        return context_tokens
    context_tokens[word_tokens_idx[0] - 1] = tokenizer.mask_token_id
    return context_tokens


def mask_next_token(context_tokens, word_tokens_idx, tokenizer):
    if len(word_tokens_idx) == 0:
        return context_tokens
    context_tokens[word_tokens_idx[-1] + 1] = tokenizer.mask_token_id
    return context_tokens


def mask_previous_and_next_token(context_tokens, word_tokens_idx, tokenizer):
    context_tokens = mask_previous_token(context_tokens, word_tokens_idx, tokenizer.mask_token_id)
    context_tokens = mask_next_token(context_tokens, word_tokens_idx, tokenizer.mask_token_id)
    return context_tokens


def word_and_mask(context_tokens, word_tokens_idx, tokenizer):
    original_context_len = context_tokens.shape[0]
    last_word_token = word_tokens_idx[-1] if len(word_tokens_idx) > 0 else torch.sum(context_tokens > 0) - 2
    context_tokens = torch.cat((
        context_tokens[:last_word_token + 1],
        tokenizer.encode('and', add_special_tokens=False, return_tensors='pt')[0],
        torch.tensor([tokenizer.mask_token_id]),
        context_tokens[last_word_token + 1:]))[:original_context_len]
    return context_tokens


def mask_and_word(context_tokens, word_tokens_idx, tokenizer):
    original_context_len = context_tokens.shape[0]
    first_word_token = word_tokens_idx[0] if len(word_tokens_idx) > 0 else 1
    context_tokens = torch.cat((
        context_tokens[:first_word_token],
        torch.tensor([tokenizer.mask_token_id]),
        tokenizer.encode('and', add_special_tokens=False, return_tensors='pt')[0],
        context_tokens[first_word_token:]))[:original_context_len]
    return context_tokens


def mask_each_token_at_once(context_tokens: torch.Tensor, word_tokens_idx, tokenizer):
    for token_id in word_tokens_idx:
        result = context_tokens.clone()  # maybe there should be .detach(), idk
        result[token_id] = tokenizer.mask_token_id
        yield result


def dont_mask(context_tokens, word_tokens_idx, mask_token_id):
    return context_tokens


CONTEXT_MASKERS = {
    'dont_mask': dont_mask,
    'first_token': mask_first_token,
    'all_word_tokens': mask_all_word_tokens,
    'previous_token': mask_previous_token,
    'next_token': mask_next_token,
    'previous_and_next_token': mask_previous_and_next_token,
    'word_and_mask': word_and_mask,
    "mask_and_word": mask_and_word
}
