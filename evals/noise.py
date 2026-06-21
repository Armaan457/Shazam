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

def create_noisy_query(
    audio_path: Path,
    *,
    sr: int,
    clip_start_s: float,
    clip_duration_s: float,
    noise_factor: float = 0.0,
):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    start = int(clip_start_s * sr)
    end = start + int(clip_duration_s * sr)
    clip = y[start:end]

    if noise_factor > 0:
        clip = add_noise(clip, noise_factor)

    temp_dir = tempfile.mkdtemp()
    out_path = Path(temp_dir) / "query.wav"
    sf.write(out_path, clip, sr)
    return out_path