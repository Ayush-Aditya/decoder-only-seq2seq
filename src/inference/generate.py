import torch


def greedy_decode(model, input_ids, max_len, pad_token_id, bos_token_id, eos_token_id):
    model.eval()

    batch_size = input_ids.size(0)
    generated_target = torch.full(
        (batch_size, 1),
        bos_token_id,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    finished = torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool)

    for _ in range(max_len):
        decoder_input = torch.cat([input_ids, generated_target], dim=1)
        attention_mask = (decoder_input != pad_token_id).long()

        logits = model(decoder_input, attention_mask=attention_mask)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated_target = torch.cat([generated_target, next_token], dim=1)

        finished = finished | (next_token.squeeze(1) == eos_token_id)
        if finished.all():
            break

    return generated_target