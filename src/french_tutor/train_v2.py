"""HF Trainer construction for wav2vec2-CTC fine-tuning — corrected rewrite.

Fixes vs. v1:
- Use processing_class=processor (current API) instead of tokenizer=processor.feature_extractor (deprecated)
- ddp_find_unused_parameters=True is hard-required for freeze_feature_encoder + DDP
- compute_metrics does not mutate label_ids in place
- gradient_checkpointing must be False under DDP with frozen params (asserted)
- Fail fast if config is missing required keys
"""
from __future__ import annotations

from typing import Any

import numpy as np
from jiwer import wer


def compute_metrics_factory(processor: Any):
    """Returns compute_metrics(eval_pred) -> {"wer": float}.

    Does NOT mutate eval_pred arrays (v1 mutated label_ids in place).
    """

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        predicted_ids = np.argmax(logits, axis=-1)
        label_ids_clean = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        predictions = processor.batch_decode(predicted_ids)
        references = processor.batch_decode(label_ids_clean, group_tokens=False)
        return {"wer": wer(references, predictions)}

    return compute_metrics


def _validate_train_cfg(train_cfg: dict[str, Any]) -> None:
    required = [
        "output_dir", "num_train_epochs", "per_device_train_batch_size",
        "per_device_eval_batch_size", "gradient_accumulation_steps", "learning_rate",
        "warmup_ratio", "weight_decay", "fp16", "gradient_checkpointing",
        "eval_strategy", "save_strategy", "logging_steps",
        "load_best_model_at_end", "seed", "report_to",
    ]
    missing = [k for k in required if k not in train_cfg]
    if missing:
        raise KeyError(f"train config missing required keys: {missing}")

    if train_cfg["gradient_checkpointing"]:
        raise ValueError(
            "gradient_checkpointing must be False under DDP with frozen feature encoder. "
            "PyTorch DDP cannot reduce gradients twice for the same parameter, which "
            "happens when checkpointing wraps frozen modules."
        )


def build_trainer(
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any,
    train_cfg: dict[str, Any],
    sampling_cfg: dict[str, Any],
):
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    from french_tutor.data_v2 import DataCollatorCTCWithPadding

    _validate_train_cfg(train_cfg)
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    kwargs = dict(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        logging_steps=train_cfg["logging_steps"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        seed=train_cfg["seed"],
        report_to=train_cfg["report_to"],
        max_steps=sampling_cfg.get("max_steps") or -1,
        ddp_find_unused_parameters=True,
    )
    training_args = TrainingArguments(**kwargs)

    callbacks = []
    if train_cfg["eval_strategy"] != "no":
        patience = train_cfg.get("early_stopping_patience", 3)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=collator,
        compute_metrics=compute_metrics_factory(processor) if eval_dataset is not None else None,
        callbacks=callbacks,
    )
