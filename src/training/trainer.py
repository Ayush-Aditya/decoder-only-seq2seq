import torch
import torch.nn.functional as F

from torch import nn
from torch.optim import Optimizer

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ProgressBar

from src.training.utils import build_decoder_seq2seq_batch


def create_trainer(
    model: nn.Module,
    optimizer: Optimizer,
    pad_token_id: int,
    device: torch.device,
) -> Engine:
    model.to(device)

    def train_step(engine, batch):
        model.train()

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        decoder_inputs, labels, attention_mask = build_decoder_seq2seq_batch(
            input_ids=input_ids,
            target_ids=target_ids,
            pad_token_id=pad_token_id,
        )

        logits = model(decoder_inputs, attention_mask=attention_mask)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_token_id,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    trainer = Engine(train_step)

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    pbar = ProgressBar()
    pbar.attach(trainer, ["loss"])

    return trainer

