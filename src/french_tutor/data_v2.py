"""DataCollatorCTCWithPadding — corrected rewrite.

Fixes vs. v1:
- No as_target_processor() context manager (deprecated, removed in transformers v5)
- Use processor.tokenizer.pad() directly for label padding
- Validate inputs aren't empty
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("DataCollatorCTCWithPadding received empty batch")

        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
