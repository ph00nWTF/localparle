.PHONY: setup data baseline train eval cli gradio tutor-text serve pull-mistral pull-pi test lint

setup:
	uv sync --extra dev
	@uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('CUDA OK:', torch.cuda.get_device_name(0))"
	@command -v ffmpeg >/dev/null || (echo "ffmpeg missing" && exit 1)
	@ollama list | grep -q mistral || ollama pull mistral

data:
	uv run python scripts/prepare_data.py --config configs/data.yaml

baseline:
	uv run python scripts/train.py --config configs/train.yaml --baseline

train:
	uv run python scripts/train.py --config configs/train.yaml

eval:
	uv run python scripts/evaluate.py --config configs/train.yaml

cli:
	uv run python scripts/tutor_cli.py --config configs/tutor.yaml

gradio:
	uv run python scripts/tutor_gradio.py --config configs/tutor.yaml

tutor-text:
	uv run python scripts/tutor_text.py --config configs/tutor.yaml

serve:
	uv run python scripts/serve_web.py --config configs/tutor.yaml

pull-mistral:
	ollama pull mistral-nemo

pull-pi:
	ollama pull gemma3:1b

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts
	uv run mypy src
