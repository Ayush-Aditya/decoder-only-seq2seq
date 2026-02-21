import torch
from torch.nn.utils.rnn import pad_sequence


def shift_for_teacher_forcing(input_ids: torch.Tensor, pad_token_id: int):
    """
    Prepare inputs and labels for decoder-only next-token prediction.

    Example:
    input:  [SOS, a, b, c, EOS]
    inputs: [SOS, a, b, c]
    labels: [a,   b, c, EOS]
    """

    inputs = input_ids[:, :-1].contiguous()
    labels = input_ids[:, 1:].contiguous()

    labels = labels.clone()
    labels[labels == pad_token_id] = pad_token_id

    return inputs, labels


def build_decoder_seq2seq_batch(
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    pad_token_id: int,
):
    """
    Build decoder-only seq2seq teacher-forcing tensors.

    Decoder input is:
        [input_sequence, target_sequence_without_last_token]

    Labels are:
        [ignore_for_all_input_positions, target_sequence_shifted_left]

    This trains the model to predict only target tokens while conditioning
    on the full input prefix.
    """

    decoder_inputs_list = []
    labels_list = []

    batch_size = input_ids.size(0)
    for sample_idx in range(batch_size):
        input_len = (input_ids[sample_idx] != pad_token_id).sum().item()
        target_len = (target_ids[sample_idx] != pad_token_id).sum().item()

        source_tokens = input_ids[sample_idx, :input_len]
        target_tokens = target_ids[sample_idx, :target_len]

        target_inputs = target_tokens[:-1]
        target_labels = target_tokens[1:]

        sample_decoder_inputs = torch.cat([source_tokens, target_inputs], dim=0)
        sample_labels = torch.cat(
            [
                torch.full((input_len,), pad_token_id, device=input_ids.device),
                target_labels,
            ],
            dim=0,
        )

        decoder_inputs_list.append(sample_decoder_inputs)
        labels_list.append(sample_labels)

    decoder_inputs = pad_sequence(
        decoder_inputs_list,
        batch_first=True,
        padding_value=pad_token_id,
    )

    labels = pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=pad_token_id,
    )

    attention_mask = (decoder_inputs != pad_token_id).long()

    return decoder_inputs, labels, attention_mask