"""
Export module: save current soundscape as WAV file.
Provides a helper to write audio data to a deterministic filename based on seed.
"""

import numpy as np
import wave
import struct
from pathlib import Path
from typing import Optional


def export_wav(
    audio: np.ndarray,
    sample_rate: int = 44100,
    filepath: Optional[str] = None,
    seed: Optional[str] = None,
) -> str:
    """
    Write a mono or stereo audio array to a WAV file.

    :param audio: 1D (mono) or 2D (channels x samples) numpy array of float32 in [-1, 1]
    :param sample_rate: sample rate in Hz
    :param filepath: explicit output path; if None, generate from seed
    :param seed: seed string used to derive filename (ignored if filepath given)
    :return: the path to the written file
    """
    if filepath is None:
        if seed is None:
            seed = "soundscape"
        # sanitize seed for filename
        safe_seed = "".join(c if c.isalnum() or c in "-_ " else "_" for c in seed).strip()
        safe_seed = safe_seed.replace(" ", "_")
        filepath = f"{safe_seed}.wav"

    path = Path(filepath)
    # ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # convert to int16
    if audio.ndim == 1:
        audio_int = np.int16(audio * 32767)
        n_channels = 1
    elif audio.ndim == 2:
        # shape (channels, samples) -> (samples, channels) for wav
        audio_int = np.int16(audio.T * 32767)
        n_channels = audio.shape[0]
    else:
        raise ValueError(f"audio array must be 1D or 2D, got shape {audio.shape}")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int.tobytes())

    return str(path.resolve())
