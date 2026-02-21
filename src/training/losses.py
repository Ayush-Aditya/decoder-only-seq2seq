import torch
import torch.nn.functional as F


def compute_loss(logits, labels, pad_token_id: int):
    """
    Cross-entropy loss ignoring padding tokens.
    """

    vocab_size = logits.size(-1)

    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    loss = F.cross_entropy(
        logits,
        labels,
        ignore_index=pad_token_id,
    )

    return loss