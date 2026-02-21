import torch


def sequence_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int):
    """
    Computes exact sequence match accuracy.
    A sequence is correct only if ALL tokens match (ignoring pad).
    """

    preds = logits.argmax(dim=-1)

    mask = labels != pad_token_id

    correct_tokens = (preds == labels) | ~mask
    correct_sequences = correct_tokens.all(dim=1)

    return correct_sequences.float().mean().item()


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int):
    """
    Computes token-level accuracy, ignoring pad tokens.
    """

    preds = logits.argmax(dim=-1)
    mask = labels != pad_token_id

    if mask.sum() == 0:
        return 0.0

    correct = (preds == labels) & mask
    return correct.float().sum().item() / mask.float().sum().item()