# Self-Assessment Rubric

Maps the four primary goals from `french_proposal.md` to verifiable evidence. Each goal is scored Met / Partial / Not met with a pointer to the artifact that proves it.

| Goal | Status | Evidence |
|---|---|---|
| **1.** Fine-tune wav2vec 2.0 on CV-FR and achieve test WER below 20%. | **Met** | `docs/results/wer_table.md` shows greedy 17.84% and beam+LM 14.86% |
| **2.** Demonstrate fine-tuning meaningfully improves over a baseline. | **Met** | `docs/results/wer_table.md` shows baseline 27.5% improving to 17.84% greedy (35% relative improvement) |
| **3.** Integrate ASR into a working real-time spoken French tutor. | **Met** | Web UI at localhost:8000, mic to ASR to Mistral to Piper TTS pipeline verified end-to-end |
| **4.** Produce training loss curves and WER evaluation metrics. | **Met** | `docs/results/loss_curves.png` and `docs/results/wer_table.md` |
| Stretch. Deploy on Raspberry Pi with AI Hat. | **Not met** | Not implemented due to time constraints |

## Evidence collection rules

- WER is computed by `jiwer.wer` on the official Common Voice test split. Test transcripts were never seen during training or model selection. Model selection used the dev split.
- Loss curves are generated from TensorBoard event files in `runs/`.
- Tutor latency is wall-clock from end-of-mic-input to start-of-audio-output, averaged over 10 turns of conversational French at the laptop.
