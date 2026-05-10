"""Build wav2vec2-CTC + processor; greedy decoding."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch


def build_processor(vocab_path: str | Path, sample_rate: int = 16000) -> Any:
    """Construct a Wav2Vec2Processor from a char-level vocab.json + feature extractor."""
    from transformers import (
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
    )

    vocab_path = Path(vocab_path)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    # These keys are set by build_vocab in data.py — read directly, don't infer by position.
    pad_token = "[PAD]"
    unk_token = "[UNK]"
    word_delimiter = "|"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        tokenizer = Wav2Vec2CTCTokenizer(
            str(tmp_path / "vocab.json"),
            unk_token=unk_token,
            pad_token=pad_token,
            word_delimiter_token=word_delimiter,
        )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sample_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def build_model(backbone: str, processor: Any, model_cfg: dict[str, Any]) -> Any:
    """Construct Wav2Vec2ForCTC with our vocab head and apply freeze flags."""
    from transformers import Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(
        backbone,
        attention_dropout=model_cfg["attention_dropout"],
        hidden_dropout=model_cfg["hidden_dropout"],
        feat_proj_dropout=model_cfg["feat_proj_dropout"],
        mask_time_prob=model_cfg["mask_time_prob"],
        layerdrop=model_cfg["layerdrop"],
        ctc_loss_reduction=model_cfg["ctc_loss_reduction"],
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )
    if model_cfg.get("freeze_feature_extractor"):
        model.freeze_feature_extractor()
    return model


def greedy_decode(logits: torch.Tensor, processor: Any) -> list[str]:
    """Argmax → CTC collapse → token-ids → strings."""
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)
