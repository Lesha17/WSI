import torch
import tqdm


def get_model_inputs(batch):
    input_keys = ['input_ids', 'attention_mask']
    return {k: batch[k] for k in input_keys}


def get_word_mask(dataset):
    return torch.stack([smpl['given_word_mask'] for smpl in dataset])


def get_word_token_ids(dataset):
    return [smpl['word_token_ids'] for smpl in dataset]


def get_attention_mask(dataset):
    return torch.stack([smpl['attention_mask'] for smpl in dataset])


def to_device(batch, device='cuda'):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# TODO add SIF
def get_target_tokens_avg_vector(bert_out, target_tokens_mask, bert_layer=-1):
    masked_out = bert_out[bert_layer] * target_tokens_mask.unsqueeze(-1)
    masked_out_sum = torch.sum(masked_out, dim=1)
    mask_sum = torch.sum(target_tokens_mask, dim=1)
    mask_sum[mask_sum == 0] = 1 # to avoid division by zero
    return masked_out_sum / mask_sum.unsqueeze(-1)


def get_given_token_vector(bert_out, target_token_id, bert_layer=-1):
    layer_bert_out = bert_out[bert_layer]
    first_token_id = torch.tensor(target_token_id)\
        .view(-1, 1)\
        .expand(-1, layer_bert_out.shape[-1])\
        .view(-1, 1, layer_bert_out.shape[-1])
    return layer_bert_out.gather(1, first_token_id).squeeze(1)


# --- word_vector_fn ---


def get_avg_word_tokens_vector(bert_out, dataset, bert_layer=-1):
    word_mask = get_word_mask(dataset)
    return get_target_tokens_avg_vector(bert_out, word_mask, bert_layer=bert_layer)


def get_avg_context_vector(bert_out, dataset, bert_layer=-1):
    attention_mask = get_attention_mask(dataset)
    return get_target_tokens_avg_vector(bert_out, attention_mask, bert_layer=bert_layer)


def get_avg_context_without_word_tokens(bert_out, dataset, bert_layer=-1):
    attention_mask = get_attention_mask(dataset)
    word_mask = get_word_mask(dataset)
    return get_target_tokens_avg_vector(bert_out, attention_mask - word_mask, bert_layer=bert_layer)


def get_first_word_token_vector(bert_out, dataset, bert_layer=-1):
    word_token_ids = get_word_token_ids(dataset)
    first_token_id = [ids[0] if len(ids) > 0 else 0 for ids in word_token_ids]
    return get_given_token_vector(bert_out, first_token_id, bert_layer=bert_layer)


def get_first_context_vector(bert_out, dataset, bert_layer=-1):
    return bert_out[bert_layer][:, 0, :]


def get_previous_token_vector(bert_out, dataset, bert_layer=-1):
    word_token_ids = get_word_token_ids(dataset)
    first_token_id = [ids[0] if len(ids) > 0 else 0 for ids in word_token_ids]
    previous_token_id = [max(i-1, 0) for i in first_token_id]
    return get_given_token_vector(bert_out, previous_token_id, bert_layer=bert_layer)


def get_next_token_vector(bert_out, dataset, bert_layer=-1):
    seq_len = torch.sum(get_attention_mask(dataset), dim=1)
    word_token_ids = get_word_token_ids(dataset)
    last_token_id = torch.tensor([ids[-1] if len(ids) > 0 else seq_len[i]-1 for i, ids in enumerate(word_token_ids)])
    next_token_id = torch.min(last_token_id + 1, seq_len - 1)
    return get_given_token_vector(bert_out, next_token_id, bert_layer=bert_layer)


def forward(batch, bertModel, bert_gradients=False):
    model_inputs = to_device(get_model_inputs(batch), device=bertModel.device)
    if bert_gradients:
        bert_out = bertModel(**model_inputs, output_hidden_states=True)
    else:
        with torch.no_grad():
            bert_out = bertModel(**model_inputs, output_hidden_states=True)
    return bert_out


def get_all_vectors(dataset, bertModel, batch_size=64, progress=False):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    result_batches = []
    if progress:
        dataloader = tqdm.tqdm(dataloader)
    for batch in dataloader:
        bert_out = forward(batch, bertModel)
        hidden_states = bert_out[2]
        for layer_num, hidden in enumerate(hidden_states):
            hidden = hidden.detach().to('cpu')
            if len(result_batches) <= layer_num:
                result_batches.append([])
            result_batches[layer_num].append(hidden)

    result = []
    for layer_batches in result_batches:
        result.append(torch.cat(layer_batches, dim=0))
    return result


WORD_VECTOR_FNS = {
    'avg_word_tokens_vector': get_avg_word_tokens_vector,
    'avg_context_vector': get_avg_context_vector,
    'avg_context_without_word_tokens': get_avg_context_without_word_tokens,
    'first_word_token_vector': get_first_word_token_vector,
    'first_context_vector': get_first_context_vector,
    'previous_token_vector': get_previous_token_vector,
    'next_token_vector': get_next_token_vector
}