Machine Learning Project Proposal
Project: Local AI French Language Tutor with Speech Recognition
Dataset
The primary dataset will be the Mozilla Common Voice French corpus, a free and open source collection of thousands of hours of labeled French speech recordings contributed by volunteers. The full dataset is approximately 60GB and will be stored and processed locally. This dataset provides the labeled audio needed to train a speech-to-text model that understands spoken French.
Modeling Approach
The project will fine-tune wav2vec 2.0, an open source speech recognition model developed by Meta/Facebook, on the French Common Voice dataset. wav2vec 2.0 is a transformer-based deep learning model that learns representations of raw audio and can be fine-tuned for automatic speech recognition (ASR) on a specific language. Training and inference will be done entirely locally using Python and PyTorch with CUDA on an NVIDIA RTX 2000 Ada Generation laptop GPU. No cloud services or OpenAI tools will be used.
Once the ASR model is trained, it will be integrated into a simple conversational tutor application. The pipeline will work as follows: the user speaks French into a microphone, the trained ASR model transcribes the speech to text, a lightweight open source large language model (such as Mistral 7B or LLaMA 3 8B running via Ollama) responds in French acting as a conversation tutor, and a text-to-speech engine speaks the response back to the user. The LLM will be prompted to correct grammar mistakes, maintain a natural French conversation, and adjust to the user's level.
Goals for the Model
The following goals will be used to develop the self-assessment rubric:
1. Successfully fine-tune wav2vec 2.0 on the Mozilla Common Voice French dataset and achieve a Word Error Rate (WER) below 20% on a held-out test set.
2. Demonstrate that fine-tuning on French data meaningfully improves transcription accuracy compared to a baseline.
3. Integrate the ASR model into a working real-time tutor application where a user can have a back-and-forth spoken conversation in French.
4. Produce training loss curves and WER evaluation metrics to document model performance.
5. Stretch goal: export and deploy the system on a Raspberry Pi with the AI Hat for a standalone physical conversation device.
All training, evaluation, and application code will be written in Python and documented for reproducibility.
