import argparse
from pathlib import Path
from typing import Optional

import torch
from torchviz import make_dot

from model.dereverberation_simple import DereverberationModel
from model.perceptual_qualitynet import PerceptualQualityNet


def _load_state_dict(checkpoint_path: Path, device: torch.device):
	checkpoint = torch.load(checkpoint_path, map_location=device)
	if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
		state_dict = checkpoint["state_dict"]
	elif isinstance(checkpoint, dict):
		state_dict = checkpoint
	else:
		raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

	return state_dict


def _try_load_with_common_prefix_fixes(model: torch.nn.Module, state_dict: dict):
	try:
		model.load_state_dict(state_dict)
		return
	except RuntimeError as first_error:
		prefixes = ("module.", "model.", "perceptual_net.")
		for prefix in prefixes:
			if any(key.startswith(prefix) for key in state_dict):
				remapped = {
					(key[len(prefix) :] if key.startswith(prefix) else key): value
					for key, value in state_dict.items()
				}
				try:
					model.load_state_dict(remapped)
					return
				except RuntimeError:
					pass
		raise RuntimeError(
			"Could not load checkpoint into selected model architecture. "
			"Tried original keys and common prefixes: module., model., perceptual_net."
		) from first_error


def _build_model(model_name: str, sample_rate: int) -> torch.nn.Module:
	if model_name == "perceptual_quality":
		return PerceptualQualityNet(sample_rate=sample_rate)
	if model_name == "dereverberation_simple":
		return DereverberationModel()
	raise ValueError(f"Unsupported model: {model_name}")


def visualize_model(
	model_name: str,
	checkpoint: Optional[Path],
	output: Path,
	graph_format: str,
	batch_size: int,
	segment_length: int,
	sample_rate: int,
	device: torch.device,
):
	model = _build_model(model_name=model_name, sample_rate=sample_rate).to(device)
	if checkpoint is not None:
		state_dict = _load_state_dict(checkpoint, device)
		_try_load_with_common_prefix_fixes(model, state_dict)
	model.eval()

	dummy_audio = torch.randn(
		batch_size, segment_length, device=device, requires_grad=True
	)

	quality_pred = model(dummy_audio)

	graph = make_dot(
		quality_pred.mean(),
		params={
			**dict(model.named_parameters()),
			"dummy_audio": dummy_audio,
		},
	)
	graph.format = graph_format

	output.parent.mkdir(parents=True, exist_ok=True)
	rendered_path = graph.render(str(output), cleanup=True)
	print(f"Graph saved to: {rendered_path}")


def parse_args():
	parser = argparse.ArgumentParser(
		prog="visualize",
		description="Load a model checkpoint and visualize it with torchviz.",
	)
	parser.add_argument(
		"--model",
		choices=["perceptual_quality", "dereverberation_simple"],
		default="perceptual_quality",
		help="Model architecture to instantiate before loading checkpoint",
	)
	parser.add_argument(
		"checkpoint",
		type=Path,
		nargs="?",
		default=None,
		help="Optional path to model checkpoint",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output file path without extension (default: plots/<checkpoint_stem>_graph)",
	)
	parser.add_argument(
		"--format",
		dest="graph_format",
		choices=["pdf", "png", "svg"],
		default="svg",
		help="Output graph format",
	)
	parser.add_argument(
		"--batch-size", type=int, default=1, help="Dummy batch size for forward pass"
	)
	parser.add_argument(
		"--segment-length",
		type=int,
		default=44100 * 4,
		help="Dummy waveform length in samples",
	)
	parser.add_argument(
		"--sample-rate",
		type=int,
		default=44100,
		help="Sample rate used to instantiate PerceptualQualityNet",
	)
	parser.add_argument(
		"--device",
		choices=["cpu", "cuda"],
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device used to load model and run dummy pass",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	checkpoint_path = None
	if args.checkpoint is not None:
		checkpoint_path = args.checkpoint.expanduser().resolve()
		if not checkpoint_path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	if args.output is None:
		base_name = (
			f"{checkpoint_path.stem}_graph" if checkpoint_path is not None else f"{args.model}_graph"
		)
		output_path = Path("plots") / base_name
	else:
		output_path = args.output

	device = torch.device(args.device)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is not available")

	visualize_model(
		model_name=args.model,
		checkpoint=checkpoint_path,
		output=output_path,
		graph_format=args.graph_format,
		batch_size=args.batch_size,
		segment_length=args.segment_length,
		sample_rate=args.sample_rate,
		device=device,
	)
