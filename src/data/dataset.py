import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class ReverseStringDataset(Dataset):
    """
    Synthetic dataset for string reversal task.

    Example:
        input  = "abc"
        target = "cba"
    """

    def __init__(
        self,
        vocab: str = "abcdefghijklmnopqrstuvwxyz",
        min_len: int = 3,
        max_len: int = 10,
        dataset_size: int = 10000,
    ):
        self.vocab = vocab
        self.min_len = min_len
        self.max_len = max_len
        self.dataset_size = dataset_size

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.tokens = [self.pad_token, self.bos_token, self.eos_token] + list(vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.samples = [self._build_example() for _ in range(self.dataset_size)]

    def __len__(self) -> int:
        return self.dataset_size

    def _generate_string(self) -> str:
        length = random.randint(self.min_len, self.max_len)
        return "".join(random.choice(self.vocab) for _ in range(length))

    def _encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def _build_example(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._generate_string()
        rev = s[::-1]

        input_tokens = (
            [self.stoi[self.bos_token]]
            + self._encode(s)
            + [self.stoi[self.eos_token]]
        )

        target_tokens = (
            [self.stoi[self.bos_token]]
            + self._encode(rev)
            + [self.stoi[self.eos_token]]
        )

        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]
    
from torch.nn.utils.rnn import pad_sequence
from typing import Dict


def collate_fn(batch) -> Dict[str, torch.Tensor]:
    """
    Pads batch of variable length sequences.

    Returns:
        dict with:
            input_ids
            target_ids
            attention_mask
    """
    input_ids, target_ids = zip(*batch)

    
    input_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)

    
    attention_mask = (input_padded != 0).long()

    return {
        "input_ids": input_padded,
        "target_ids": target_padded,
        "attention_mask": attention_mask,
    }