from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def add_noise(audio: np.ndarray, noise_factor: float = 0.02):
    noise = np.random.normal(
        0,
        np.std(audio),
        len(audio)
    )
    noisy = audio + noise_factor * noise
    return np.clip(noisy, -1.0, 1.0)

def pitch_shift_audio(
    audio: np.ndarray,
    sr: int,
    n_steps: float,
):
    return librosa.effects.pitch_shift(
        audio,
        sr=sr,
        n_steps=n_steps,
    )

def time_stretch_audio(
    audio: np.ndarray,
    rate: float,
):
    stretched = librosa.effects.time_stretch(
        audio,
        rate=rate,
    )

    return stretched

def add_reverb(
    audio: np.ndarray,
    decay: float = 0.3,
    delay_ms: float = 40,
    sr: int = 22050,
):
    delay_samples = int(delay_ms * sr / 1000)

    impulse = np.zeros(delay_samples * 4 + 1)
    impulse[0] = 1.0

    for i in range(1, 5):
        idx = i * delay_samples
        impulse[idx] = decay ** i

    reverbed = np.convolve(audio, impulse, mode="full")
    reverbed = reverbed[: len(audio)]
    return np.clip(reverbed, -1.0, 1.0)

def compress_dynamic_range(
    audio: np.ndarray,
    threshold: float = 0.3,
    ratio: float = 4.0,
):
    out = audio.copy()
    mask = np.abs(out) > threshold
    out[mask] = np.sign(out[mask]) * (
        threshold
        + (np.abs(out[mask]) - threshold) / ratio
    )
    return np.clip(out, -1.0, 1.0)


def simulate_phone_recording(
    audio: np.ndarray,
    sr: int,
):
    audio = librosa.effects.preemphasis(audio)

    audio = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=8000,
    )

    audio = librosa.resample(
        audio,
        orig_sr=8000,
        target_sr=sr,
    )

    return np.clip(audio, -1.0, 1.0)


def create_noisy_query(
    audio_path: Path,
    *,
    sr: int,
    clip_start_s: float,
    clip_duration_s: float,
    noise_factor: float = 0.0,
    pitch_shift_steps: float = 0.0,
    time_stretch_rate: float = 1.0,
    use_reverb: bool = False,
    use_phone_simulation: bool = False,
    use_compression: bool = False,
):
    y, _ = librosa.load(
        audio_path,
        sr=sr,
        mono=True,
    )

    start = int(clip_start_s * sr)
    end = start + int(clip_duration_s * sr)

    clip = y[start:end]

    if pitch_shift_steps != 0:
        clip = pitch_shift_audio(
            clip,
            sr,
            pitch_shift_steps,
        )

    if time_stretch_rate != 1.0:
        clip = time_stretch_audio(
            clip,
            time_stretch_rate,
        )

    if noise_factor > 0:
        clip = add_noise(
            clip,
            noise_factor,
        )

    if use_reverb:
        clip = add_reverb(
            clip,
            sr=sr,
        )

    if use_compression:
        clip = compress_dynamic_range(
            clip,
        )

    if use_phone_simulation:
        clip = simulate_phone_recording(
            clip,
            sr,
        )

    temp_dir = tempfile.mkdtemp()

    out_path = Path(temp_dir) / "query.wav"

    sf.write(
        out_path,
        clip,
        sr,
    )
    return out_path