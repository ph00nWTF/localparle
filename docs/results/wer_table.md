# WER Evaluation Results

Model: `wav2vec2-xls-r-300m` fine-tuned on Common Voice French 25.0 (613,420 train samples, 10 epochs).
Evaluated on the official CV-FR test split (16,201 samples). Computed with `jiwer.wer`.

## Overall

| Model | Decode | WER | CER |
|---|---|---|---|
| `wav2vec2-large-xlsr-53-french` (baseline, zero-shot) | greedy | 27.50% | — |
| **Ours (fine-tuned)** | greedy | **17.84%** | 5.18% |
| **Ours (fine-tuned)** | beam=100, KenLM 4-gram | **14.86%** | 4.47% |

Relative improvement over baseline: **35%** (greedy) / **46%** (beam+LM).

## By reference length (greedy decode)

| Bucket | n | WER |
|---|---|---|
| ≤ 5 words | 1,546 | 23.88% |
| 6–10 words | 7,210 | 17.92% |
| 11–20 words | 7,444 | 17.37% |

Longer sentences score better — more context for the CTC decoder to recover from local mistakes.

## Sample ref/hyp pairs

| Reference | Hypothesis |
|---|---|
| ce dernier a évolué tout au long de l'histoire romaine | ce dernier évolé tout au long de l'histoire romaine |
| j'ai dit que les acteurs de bois avaient selon moi beaucoup d'avantages sur les autres | j ai dit que les acteurs de bois avaient selon moi beaucoup davantage sur les autres |
| les pays-bas ont remporté toutes les éditions | le pays-bas a remporté toutes les éditions |
| à ce jour on ne sait pas précisément qui étaient les réalisateurs du film | à ce jour on ne sait pas précisément qui étaient les réalisateurs du film ✓ |

Full sample set (10 pairs) in `models/final-v2/eval_results.json`.
