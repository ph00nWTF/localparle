"""HF Trainer construction + compute_metrics for fine-tuning wav2vec2-CTC."""
from __future__ import annotations

from typing import Any

import numpy as np
from jiwer import wer


def compute_metrics_factory(processor: Any):
    """Returns a compute_metrics(eval_pred) -> {wer} callable for HF Trainer."""

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        predicted_ids = np.argmax(logits, axis=-1)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.batch_decode(predicted_ids)
        references = processor.batch_decode(label_ids, group_tokens=False)
        return {"wer": wer(references, predictions)}

    return compute_metrics


def build_trainer(
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any,
    train_cfg: dict[str, Any],
    sampling_cfg: dict[str, Any],
):
    """Construct transformers.Trainer with our collator, metrics, and callbacks."""
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    from french_tutor.data import DataCollatorCTCWithPadding

    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
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

    callbacks = []
    if train_cfg["eval_strategy"] != "no":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"]))

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=collator,
        compute_metrics=compute_metrics_factory(processor),
        callbacks=callbacks,
    )
