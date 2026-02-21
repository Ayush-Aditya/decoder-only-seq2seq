import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from ignite.engine import Events

from src.training.checkpointing import setup_checkpointing
from src.data.dataset import ReverseStringDataset, collate_fn
from src.models.decoder_transformer import DecoderOnlyTransformer
from src.training.trainer import create_trainer
from src.training.evaluator import create_evaluator

def load_config(path: str) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        fallback_path = Path("configs") / "config.yaml"
        if fallback_path.exists() and fallback_path.resolve() != config_path.resolve():
            with open(fallback_path, "r") as f:
                fallback_config = yaml.safe_load(f)
            if isinstance(fallback_config, dict):
                print(f"Using fallback config: {fallback_path}")
                return fallback_config

        raise ValueError(
            f"Config file is empty or invalid YAML: {config_path}. "
            "Provide a valid config or use configs/config.yaml"
        )

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping/object: {config_path}")

    return config


def make_experiment_dir(base_dir: str) -> Path:
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    exp_id = len(list(Path(base_dir).glob("exp_*"))) + 1
    exp_dir = Path(base_dir) / f"exp_{exp_id:03d}"
    exp_dir.mkdir()

    return exp_dir


def main(config_path: str):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = make_experiment_dir(config["experiment"]["output_dir"])
    print(f"🚀 Experiment dir: {exp_dir}")

  
    dataset = ReverseStringDataset(
        dataset_size=config["data"]["num_samples"],
        max_len=config["data"]["max_len"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )


    val_dataset = ReverseStringDataset(
        dataset_size=config["data"]["num_samples"] // 10,
        max_len=config["data"]["max_len"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )


    model = DecoderOnlyTransformer(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        max_len=config["model"]["max_len"],
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
    )


    trainer = create_trainer(
        model=model,
        optimizer=optimizer,
        pad_token_id=config["data"]["pad_token_id"],
        device=device,
    )


    evaluator = create_evaluator(
        model=model,
        pad_token_id=config["data"]["pad_token_id"],
        device=device,
    )


    ckpt_dir = exp_dir / "checkpoints"

    setup_checkpointing(
        trainer=trainer,
        model=model,
        optimizer=optimizer,
        output_dir=ckpt_dir,
    )


    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        evaluator.run(val_loader)
        val_loss = evaluator.state.metrics["loss"]
        val_seq_acc = evaluator.state.metrics["seq_acc"]
        val_tok_acc = evaluator.state.metrics["tok_acc"]

        evaluator.run(dataloader)
        train_seq_acc = evaluator.state.metrics["seq_acc"]
        train_tok_acc = evaluator.state.metrics["tok_acc"]

        print(
            f"📊 Val loss: {val_loss:.4f} | "
            f"Val seq_acc: {val_seq_acc:.4f} | "
            f"Val tok_acc: {val_tok_acc:.4f} | "
            f"Train seq_acc: {train_seq_acc:.4f} | "
            f"Train tok_acc: {train_tok_acc:.4f}"
        )


    trainer.run(
        dataloader,
        max_epochs=config["training"]["max_epochs"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    args = parser.parse_args()

    main(args.config)