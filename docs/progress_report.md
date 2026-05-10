# Progress Report

## Local AI French Language Tutor with Speech Recognition

All four primary project goals have been completed. The stretch goal was not implemented due to time constraints.

### Goals 1 and 2. Fine-Tuning and Baseline Comparison

A wav2vec2-xls-r-300m model was fine-tuned on the Mozilla Common Voice French 25.0 dataset (613,420 training samples, 10 epochs) using an NVIDIA RTX 2000 Ada GPU. The fine-tuned model achieves 17.84% WER with greedy decoding and 14.86% WER with a beam search decoder paired with a 3-gram KenLM language model built from Common Voice and French Wikipedia text. The zero-shot baseline (wav2vec2-large-xlsr-53-french) scored 27.5% WER, giving a 35% relative improvement with greedy decoding and 46% with the language model. Both results are well below the 20% target.

### Goal 3. Working Real-Time Tutor

The ASR model is integrated into a web-based conversational tutor. The pipeline runs from browser microphone capture through audio decoding, wav2vec2 transcription, Mistral-Nemo response generation via Ollama, and Piper text-to-speech playback. The tutor supports CEFR-level-adaptive prompting from A1 through C1, guided and free-conversation modes, spaced-repetition vocabulary review, and real-time grammar error feedback. A second ASR pass using faster-whisper was added to handle A1 learners who respond in English.

### Goal 4. Training Metrics

Training loss curves were captured in TensorBoard and are available in the repository. WER was evaluated using jiwer on the official held-out test split (16,201 samples not seen during training or model selection). Full results are in docs/results/wer_table.md.

### Stretch Goal. Raspberry Pi Deployment

Not implemented due to time constraints.
