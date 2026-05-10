"""Test vad_trim removes leading/trailing silence and preserves speech."""
from __future__ import annotations

import io

import numpy as np
import pytest

from french_tutor.audio_io import vad_trim

SR = 16000


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def _voiced_burst(seconds: float, freq: float = 200.0, amp: float = 0.4) -> np.ndarray:
    """Synthesize a tone as a stand-in for speech. webrtcvad keys on energy + spectral
    properties of voiced sounds; a 200 Hz tone (typical male F0) at moderate amplitude
    triggers VAD reliably.
    """
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False, dtype=np.float32)
    return amp * np.sin(2 * np.pi * freq * t)


def test_vad_trim_strips_leading_and_trailing_silence():
    sig = np.concatenate([_silence(1.0), _voiced_burst(1.0), _silence(1.0)])
    trimmed = vad_trim(sig, sample_rate=SR, aggressiveness=2)
    # Original is 3s; trimmed should be roughly 1s + 200ms padding on each side ≤ 1.5s
    assert len(trimmed) < len(sig)
    assert len(trimmed) >= int(SR * 0.5)


def test_vad_trim_returns_unchanged_when_only_silence():
    sig = _silence(2.0)
    trimmed = vad_trim(sig, sample_rate=SR, aggressiveness=2)
    assert np.array_equal(trimmed, sig)


def test_vad_trim_short_audio_returns_unchanged():
    sig = _voiced_burst(0.01)  # < 30 ms frame
    trimmed = vad_trim(sig, sample_rate=SR, aggressiveness=2)
    assert np.array_equal(trimmed, sig)


def test_vad_trim_rejects_invalid_sample_rate():
    with pytest.raises(ValueError):
        vad_trim(np.zeros(1000, dtype=np.float32), sample_rate=22050)


def test_vad_trim_int16_input():
    """vad_trim should accept int16 PCM directly without re-quantizing."""
    sig = np.concatenate([_silence(0.5), _voiced_burst(0.5), _silence(0.5)])
    sig_i16 = (sig * 32768).astype(np.int16)
    trimmed = vad_trim(sig_i16, sample_rate=SR, aggressiveness=2)
    assert trimmed.dtype == np.int16
    assert len(trimmed) < len(sig_i16)


def test_preprocess_audio_normalizes_peak():
    from french_tutor.audio_io import preprocess_audio

    quiet = _voiced_burst(1.0, amp=0.01)
    out = preprocess_audio(quiet, sample_rate=SR, target_peak=0.9)
    assert abs(np.max(np.abs(out)) - 0.9) < 0.05


def test_preprocess_audio_attenuates_low_frequency_rumble():
    """High-pass should preserve speech-band content while attenuating sub-80 Hz rumble.

    Mix a 30 Hz rumble with a 200 Hz "speech" tone at equal amplitude. After the high-pass,
    the speech band must dominate (signal energy should be concentrated near 200 Hz, not 30 Hz).
    Verify by comparing pre/post RMS of low vs mid band via FFT magnitudes.
    """
    from french_tutor.audio_io import preprocess_audio

    rumble = _voiced_burst(1.0, freq=30.0, amp=0.5)
    speech = _voiced_burst(1.0, freq=200.0, amp=0.5)
    mix = rumble + speech

    out = preprocess_audio(mix, sample_rate=SR, highpass_hz=80.0, target_peak=0.9)

    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / SR)
    low = spec[(freqs >= 20) & (freqs <= 50)].sum()
    mid = spec[(freqs >= 180) & (freqs <= 220)].sum()
    assert mid > low * 5, f"high-pass did not attenuate rumble: low={low:.1f} mid={mid:.1f}"


def test_preprocess_audio_handles_empty_input():
    from french_tutor.audio_io import preprocess_audio

    out = preprocess_audio(np.array([], dtype=np.float32), sample_rate=SR)
    assert out.size == 0
