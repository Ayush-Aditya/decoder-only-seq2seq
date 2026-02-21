import argparse
from pathlib import Path

import torch
import yaml

from src.data.dataset import ReverseStringDataset
from src.inference.generate import greedy_decode
from src.models.decoder_transformer import DecoderOnlyTransformer


def load_config(config_path: str) -> dict:
	path = Path(config_path)
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {path}")

	with open(path, "r", encoding="utf-8") as file:
		cfg = yaml.safe_load(file)

	if not isinstance(cfg, dict):
		raise ValueError(f"Invalid config format in {path}")

	return cfg


def load_model_from_checkpoint(config: dict, checkpoint_path: str, device: torch.device):
	model = DecoderOnlyTransformer(
		vocab_size=config["model"]["vocab_size"],
		d_model=config["model"]["d_model"],
		nhead=config["model"]["nhead"],
		num_layers=config["model"]["num_layers"],
		max_len=config["model"]["max_len"],
	).to(device)

	checkpoint = torch.load(checkpoint_path, map_location=device)

	if isinstance(checkpoint, dict) and "model" in checkpoint:
		state_dict = checkpoint["model"]
	else:
		state_dict = checkpoint

	model.load_state_dict(state_dict)
	model.eval()
	return model


def decode_ids(token_ids, itos, bos_id, eos_id, pad_id):
	chars = []
	for token_id in token_ids:
		token_id = int(token_id)
		if token_id in (bos_id, pad_id):
			continue
		if token_id == eos_id:
			break
		chars.append(itos[token_id])
	return "".join(chars)


def encode_text(text, stoi, bos_id, eos_id):
	unknown_chars = [char for char in text if char not in stoi]
	if unknown_chars:
		chars = "".join(sorted(set(unknown_chars)))
		raise ValueError(f"Input contains unsupported chars: {chars}")

	ids = [bos_id] + [stoi[char] for char in text] + [eos_id]
	return torch.tensor(ids, dtype=torch.long)


def run_random_eval(model, dataset, device, num_samples: int):
	bos_id = dataset.stoi[dataset.bos_token]
	eos_id = dataset.stoi[dataset.eos_token]
	pad_id = dataset.stoi[dataset.pad_token]

	correct = 0
	print("\n=== Random Samples ===")
	for sample_index in range(num_samples):
		input_ids, target_ids = dataset[sample_index]

		predicted_ids = greedy_decode(
			model,
			input_ids.unsqueeze(0).to(device),
			max_len=dataset.max_len + 2,
			pad_token_id=pad_id,
			bos_token_id=bos_id,
			eos_token_id=eos_id,
		)[0].detach().cpu()

		input_text = decode_ids(input_ids.tolist(), dataset.itos, bos_id, eos_id, pad_id)
		target_text = decode_ids(target_ids.tolist(), dataset.itos, bos_id, eos_id, pad_id)
		predicted_text = decode_ids(predicted_ids.tolist(), dataset.itos, bos_id, eos_id, pad_id)

		is_correct = predicted_text == target_text
		correct += int(is_correct)

		print(
			f"sample={sample_index} | input={input_text} | target={target_text} | "
			f"pred={predicted_text} | ok={is_correct}"
		)

	print(f"random_seq_acc={correct / num_samples:.4f} ({correct}/{num_samples})")


def run_text_eval(model, dataset, device, text: str):
	bos_id = dataset.stoi[dataset.bos_token]
	eos_id = dataset.stoi[dataset.eos_token]
	pad_id = dataset.stoi[dataset.pad_token]

	input_ids = encode_text(text, dataset.stoi, bos_id, eos_id)
	predicted_ids = greedy_decode(
		model,
		input_ids.unsqueeze(0).to(device),
		max_len=dataset.max_len + 2,
		pad_token_id=pad_id,
		bos_token_id=bos_id,
		eos_token_id=eos_id,
	)[0].detach().cpu()

	predicted_text = decode_ids(predicted_ids.tolist(), dataset.itos, bos_id, eos_id, pad_id)
	expected_text = text[::-1]

	print("\n=== Custom Text ===")
	print(f"input={text}")
	print(f"target={expected_text}")
	print(f"pred={predicted_text}")
	print(f"ok={predicted_text == expected_text}")


def main():
	parser = argparse.ArgumentParser(description="Run inference checks for reverse-string model")
	parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
	parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
	parser.add_argument("--samples", type=int, default=10, help="Number of random test samples")
	parser.add_argument("--text", type=str, default="", help="Optional custom input text")
	args = parser.parse_args()

	config = load_config(args.config)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = load_model_from_checkpoint(config, args.checkpoint, device)

	dataset = ReverseStringDataset(
		dataset_size=max(args.samples, 1),
		max_len=config["data"]["max_len"],
	)

	print(f"device={device}")
	print(f"checkpoint={args.checkpoint}")

	run_random_eval(model, dataset, device, args.samples)

	if args.text:
		run_text_eval(model, dataset, device, args.text)


if __name__ == "__main__":
	main()
