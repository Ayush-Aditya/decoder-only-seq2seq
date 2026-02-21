import torch
import torch.nn.functional as F

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.handlers import ProgressBar

from src.training.utils import build_decoder_seq2seq_batch
from src.training.metrics import sequence_accuracy, token_accuracy


def create_evaluator(
    model,
    pad_token_id: int,
    device: torch.device,
):
    model.to(device)

    def eval_step(engine, batch):
        model.eval()

        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            decoder_inputs, labels, attention_mask = build_decoder_seq2seq_batch(
                input_ids=input_ids,
                target_ids=target_ids,
                pad_token_id=pad_token_id,
            )

            logits = model(decoder_inputs, attention_mask=attention_mask)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=pad_token_id,
            )

            seq_acc = sequence_accuracy(logits, labels, pad_token_id)
            tok_acc = token_accuracy(logits, labels, pad_token_id)

        return {"loss": loss.item(), "seq_acc": seq_acc, "tok_acc": tok_acc}

    evaluator = Engine(eval_step)

    RunningAverage(output_transform=lambda x: x["loss"]).attach(evaluator, "loss")
    RunningAverage(output_transform=lambda x: x["seq_acc"]).attach(
        evaluator, "seq_acc"
    )
    RunningAverage(output_transform=lambda x: x["tok_acc"]).attach(
        evaluator, "tok_acc"
    )

    ProgressBar().attach(evaluator, ["loss", "seq_acc", "tok_acc"])

    return evaluator