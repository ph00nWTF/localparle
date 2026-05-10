"""Bar chart of WER and CER across baseline, greedy, and beam+LM. Saves PNG to docs/results/."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parents[1] / "docs" / "results" / "wer_cer_comparison.png"

conditions = ["Baseline", "Fine-tuned (greedy)", "Fine-tuned (beam+LM)"]
wer = [27.50, 17.84, 14.86]
cer = [np.nan, 5.18, 4.47]
target = 20.0

x = np.arange(len(conditions))
w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_wer = ax.bar(x - w / 2, wer, w, label="WER", color="#3b6ea5")
bars_cer = ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in cer], w, label="CER", color="#e0a458")

ax.axhline(target, linestyle="--", color="crimson", linewidth=1.2)
ax.text(len(conditions) - 0.5, target + 0.6, f"Target WER {target:.0f}%", color="crimson", ha="right", fontsize=9)

for bar, v in zip(bars_wer, wer, strict=True):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.2f}%", ha="center", fontsize=9)
for bar, v in zip(bars_cer, cer, strict=True):
    if np.isnan(v):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5, "n/a", ha="center", fontsize=9, color="gray")
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.2f}%", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel("Error Rate (%)")
ax.set_title("French ASR Error Rates on Common Voice Test Split (16,201 samples)")
ax.set_ylim(0, max(wer) * 1.18)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle=":", alpha=0.5)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
