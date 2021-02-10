import torch
import tqdm


def get_model_inputs(batch):
    input_keys = ['input_ids', 'attention_mask']
    return {k: batch[k] for k in input_keys}


def to_device(batch, device='cuda'):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def get_word_vector_avg(bert_out, given_word_mask, bert_layer=-1):
    masked_out = bert_out[bert_layer] * given_word_mask.unsqueeze(-1)
    masked_out_sum = torch.sum(masked_out, dim=1)
    mask_sum = torch.sum(given_word_mask, dim=1)
    mask_sum[mask_sum == 0] = 1 # to avoid division by zero
    return masked_out_sum / mask_sum.unsqueeze(-1)


def get_word_vector_first(bert_out, given_word_mask, bert_layer=-1):
    device = bert_out[bert_layer].device
    mask_order = torch.arange(0, given_word_mask.numel(), dtype=torch.int, device=device).reshape(given_word_mask.shape)
    mask_order[given_word_mask == 0] = given_word_mask.numel()
    first_token_idx = torch.min(mask_order, axis=1).indices.unsqueeze(-1)
    bert_hidden = bert_out[bert_layer]
    indices = first_token_idx.repeat(1, bert_hidden.shape[-1]).unsqueeze(1)
    return bert_hidden.gather(1, indices).squeeze(1)


def get_word_vector_last(bert_out, given_word_mask, bert_layer=-1):
    device = bert_out[bert_layer].device
    mask_order = torch.arange(0, given_word_mask.numel(), dtype=torch.int, device=device).reshape(given_word_mask.shape)
    mask_order[given_word_mask == 0] = -1
    last_token_idx = torch.max(mask_order, axis=1).indices.unsqueeze(-1)
    bert_hidden = bert_out[bert_layer]
    indices = last_token_idx.repeat(1, bert_hidden.shape[-1]).unsqueeze(1)
    return bert_hidden.gather(1, indices).squeeze(1)


def forward(batch, bertModel, bert_gradients=False):
    batch = to_device(batch, device=bertModel.device)
    model_inputs = get_model_inputs(batch)
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
            hidden = hidden.to('cpu')
            if len(result_batches) <= layer_num:
                result_batches.append([])
            result_batches[layer_num].append(hidden)

    result = []
    for layer_batches in result_batches:
        result.append(torch.cat(layer_batches, dim=0))
    return result


def get_vectors(bert_out, given_word_mask, bert_layer=-1, word_vector_fn=get_word_vector_avg):
    return word_vector_fn(bert_out, given_word_mask, bert_layer=bert_layer, device=bert_out[bert_layer].device)