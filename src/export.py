"""
Export soundscape as WAV file.
"""

import numpy as np
import wave
import struct
from pathlib import Path
from typing import Optional


def export_wav(
    audio: np.ndarray,
    filepath: str,
    sample_rate: int = 44100,
    max_duration: Optional[float] = None,
) -> None:
    """
    Export a stereo audio array to a WAV file.

    :param audio: numpy array of shape (num_samples, 2) for stereo
    :param filepath: output file path
    :param sample_rate: sample rate in Hz
    :param max_duration: optional maximum duration in seconds
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        audio = np.column_stack((audio, audio))
    elif audio.ndim == 2 and audio.shape[1] == 1:
        audio = np.hstack((audio, audio))
    elif audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Audio must be mono or stereo (shape (N, 2))")

    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples, :]

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(filepath), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f"Exported soundscape to {filepath}