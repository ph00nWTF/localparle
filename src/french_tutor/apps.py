"""CLI loop and Gradio UI — both thin wrappers over ConversationPipeline."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from french_tutor.audio_io import play_wav_bytes, preprocess_audio, record_until_silence
from french_tutor.tutor import build_pipeline

log = logging.getLogger("apps")


def run_cli(tutor_cfg: dict[str, Any] | None = None) -> None:
    """Push-to-enter loop in the terminal: press Enter to record, speak, pause to stop, hear reply."""
    if tutor_cfg is None:
        raise ValueError("run_cli requires a config dict (configs/tutor.yaml)")

    audio_cfg = tutor_cfg.get("audio_io", {})
    sample_rate = int(audio_cfg.get("input_sample_rate", 16000))
    vad_aggressiveness = int(audio_cfg.get("vad_aggressiveness", 2))
    silence_timeout_ms = int(audio_cfg.get("silence_timeout_ms", 800))
    max_recording_s = float(audio_cfg.get("max_recording_s", 30))

    print("Building tutor pipeline (loading model…)…")
    pipeline = build_pipeline(tutor_cfg)
    print("Ready. Press Enter to start speaking. Ctrl+C to quit. 'reset' then Enter to clear history.\n")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd == "reset":
            pipeline.reset()
            print("[history cleared]\n")
            continue
        if cmd in ("quit", "exit"):
            break

        print("Speak now (pause to stop)…")
        audio = record_until_silence(
            sample_rate=sample_rate,
            vad_aggressiveness=vad_aggressiveness,
            silence_timeout_ms=silence_timeout_ms,
            max_recording_s=max_recording_s,
        )
        if len(audio) == 0:
            print("[no speech detected — try again]\n")
            continue

        audio = preprocess_audio(audio, sample_rate=sample_rate)
        result = pipeline.turn(audio, sample_rate=sample_rate)
        print(f"You: {result.transcript}")
        print(f"Tutor: {result.response_text}\n")
        play_wav_bytes(result.response_wav)


def run_gradio(tutor_cfg: dict[str, Any] | None = None) -> None:
    """gr.Blocks app: mic -> pipeline.turn -> autoplay TTS + chat transcript."""
    if tutor_cfg is None:
        raise ValueError("run_gradio requires a config dict (configs/tutor.yaml)")

    import io

    import gradio as gr
    import soundfile as sf

    from french_tutor.audio_io import vad_trim

    ui_cfg = tutor_cfg.get("ui", {})
    audio_cfg = tutor_cfg.get("audio_io", {})
    sample_rate = int(audio_cfg.get("input_sample_rate", 16000))

    print("Building tutor pipeline (loading model…)…")
    pipeline = build_pipeline(tutor_cfg)
    print("Pipeline ready. Launching Gradio…")

    opener = pipeline.scenario.get("opening_line", "")
    initial_chat = [{"role": "assistant", "content": opener}] if opener else []
    if opener:
        pipeline.history.append({"role": "assistant", "content": opener})

    def on_audio(audio: tuple[int, np.ndarray] | None, chat_history: list):
        if audio is None:
            return chat_history, None, ""
        sr_in, wave = audio
        # Gradio gives int16 PCM at the device's sample rate; resample to model SR
        wave_f32 = wave.astype(np.float32) / 32768.0 if wave.dtype == np.int16 else wave.astype(np.float32)
        if wave_f32.ndim == 2:
            wave_f32 = wave_f32.mean(axis=1)
        if sr_in != sample_rate:
            import librosa
            wave_f32 = librosa.resample(wave_f32, orig_sr=sr_in, target_sr=sample_rate)
        wave_f32 = vad_trim(wave_f32, sample_rate=sample_rate,
                            aggressiveness=int(audio_cfg.get("vad_aggressiveness", 2)))
        if len(wave_f32) < sample_rate * 0.2:
            return chat_history, None, "[too short / no speech]"

        wave_f32 = preprocess_audio(wave_f32, sample_rate=sample_rate)
        result = pipeline.turn(wave_f32, sample_rate=sample_rate)
        chat_history = chat_history + [
            {"role": "user", "content": result.transcript},
            {"role": "assistant", "content": result.response_text},
        ]
        reply_data, reply_sr = sf.read(io.BytesIO(result.response_wav), dtype="int16")
        return chat_history, (reply_sr, reply_data), ""

    def on_reset():
        pipeline.reset()
        if opener:
            pipeline.history.append({"role": "assistant", "content": opener})
        return list(initial_chat), None, "[history cleared]"

    with gr.Blocks(title="French Tutor") as demo:
        gr.Markdown("# French Tutor\nSpeak in French. The tutor will respond and gently correct mistakes.")
        chat = gr.Chatbot(type="messages", height=400, value=initial_chat)
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="numpy", label="Hold to speak")
            reply = gr.Audio(label="Tutor reply", autoplay=True, type="numpy")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            reset_btn = gr.Button("Reset conversation")
        send_btn.click(on_audio, inputs=[mic, chat], outputs=[chat, reply, status])
        reset_btn.click(on_reset, outputs=[chat, reply, status])

    demo.launch(
        server_name=ui_cfg.get("gradio_host", "127.0.0.1"),
        server_port=int(ui_cfg.get("gradio_port", 7860)),
        show_api=False,
    )
