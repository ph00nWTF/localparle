"""Microphone capture (sounddevice) + webrtcvad silence detection + playback.

Two entry points:
- `vad_trim(audio, sample_rate)`: pure-numpy silence trimmer — no mic required, testable.
- `record_until_silence(...)`: blocking mic capture that stops after a silence timeout.
"""
from __future__ import annotations

from typing import Any

import numpy as np


_VAD_FRAME_MS = 30  # webrtcvad supports 10/20/30 ms; 30 = best speech/noise discrimination
_VAD_VALID_RATES = (8000, 16000, 32000, 48000)


def _to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float [-1, 1] or int16 input to contiguous int16 PCM."""
    if audio.dtype == np.int16:
        return np.ascontiguousarray(audio)
    if audio.dtype.kind == "f":
        return np.ascontiguousarray((audio * 32768.0).clip(-32768, 32767).astype(np.int16))
    raise TypeError(f"unsupported audio dtype: {audio.dtype}")


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    target_peak: float = 0.9,
    highpass_hz: float = 80.0,
) -> np.ndarray:
    """Light audio cleanup before VAD/ASR.

    1. High-pass at `highpass_hz` to remove mic rumble / AC hum (single-pole IIR; no scipy).
    2. Peak-normalize to `target_peak` so quiet inputs don't underdrive the model.

    Returns float32 in [-1, 1]. Idempotent on near-silence (returns unchanged).
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / (32768.0 if audio.dtype == np.int16 else 1.0)

    if audio.size == 0:
        return audio

    # 4th-order Butterworth high-pass (-24 dB/octave) applied with filtfilt for zero phase.
    from scipy.signal import butter, filtfilt
    nyq = sample_rate / 2.0
    sos_b, sos_a = butter(N=4, Wn=highpass_hz / nyq, btype="highpass")
    y = filtfilt(sos_b, sos_a, audio).astype(np.float32)

    peak = float(np.max(np.abs(y)))
    if peak > 1e-4:
        y *= target_peak / peak
    return y


def vad_trim(
    audio: np.ndarray,
    sample_rate: int = 16000,
    aggressiveness: int = 2,
    pad_ms: int = 200,
) -> np.ndarray:
    """Trim leading/trailing silence from a waveform using webrtcvad.

    Returns the original audio unchanged if no speech is detected.
    `pad_ms` keeps a small margin around speech to avoid clipping word boundaries.
    """
    if sample_rate not in _VAD_VALID_RATES:
        raise ValueError(f"sample_rate must be one of {_VAD_VALID_RATES}, got {sample_rate}")
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    pcm = _to_int16(audio)

    frame_size = int(sample_rate * _VAD_FRAME_MS / 1000)
    n_frames = len(pcm) // frame_size
    if n_frames == 0:
        return audio

    is_speech = [
        vad.is_speech(pcm[i * frame_size : (i + 1) * frame_size].tobytes(), sample_rate)
        for i in range(n_frames)
    ]
    if not any(is_speech):
        return audio

    first = is_speech.index(True)
    last = n_frames - 1 - is_speech[::-1].index(True)

    pad_frames = pad_ms // _VAD_FRAME_MS
    first = max(0, first - pad_frames)
    last = min(n_frames - 1, last + pad_frames)
    return audio[first * frame_size : (last + 1) * frame_size]


def record_until_silence(
    *,
    sample_rate: int = 16000,
    vad_aggressiveness: int = 2,
    silence_timeout_ms: int = 700,
    max_recording_s: float = 30.0,
) -> np.ndarray:
    """Record from default mic; stop after `silence_timeout_ms` of silence post-speech.

    Returns float32 audio in [-1, 1]. Returns empty array if no speech detected before timeout.
    """
    if sample_rate not in _VAD_VALID_RATES:
        raise ValueError(f"sample_rate must be one of {_VAD_VALID_RATES}, got {sample_rate}")

    import sounddevice as sd
    import webrtcvad

    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_size = int(sample_rate * _VAD_FRAME_MS / 1000)
    silence_threshold = silence_timeout_ms // _VAD_FRAME_MS
    max_frames = int(max_recording_s * 1000 / _VAD_FRAME_MS)

    chunks: list[np.ndarray] = []
    silence_count = 0
    seen_speech = False

    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="int16", blocksize=frame_size
    ) as stream:
        for _ in range(max_frames):
            frame, _overflowed = stream.read(frame_size)
            frame = frame.flatten()
            chunks.append(frame)
            if vad.is_speech(frame.tobytes(), sample_rate):
                seen_speech = True
                silence_count = 0
            elif seen_speech:
                silence_count += 1
                if silence_count >= silence_threshold:
                    break

    if not seen_speech:
        return np.array([], dtype=np.float32)

    pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
    return vad_trim(pcm, sample_rate=sample_rate, aggressiveness=vad_aggressiveness)


def play_wav_bytes(wav_bytes: bytes) -> None:
    """Decode wav bytes and play through default output device (blocking)."""
    import io
    import soundfile as sf
    import sounddevice as sd

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, sr)
    sd.wait()


def list_devices() -> Any:
    """Return sounddevice.query_devices() — used by scripts to confirm mic/speaker presence."""
    import sounddevice as sd

    return sd.query_devices()
