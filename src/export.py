"""Export soundscape audio stream to WAV file."""

import numpy as np
import wave
import struct
from typing import Optional


def export_to_wav(
    audio: np.ndarray,
    filepath: str,
    sample_rate: int = 44100,
    duration: Optional[float] = None,
) -> None:
    """
    Export a numpy audio array to a WAV file.

    :param audio: 2D numpy array of shape (num_samples, num_channels)
    :param filepath: output .wav file path
    :param sample_rate: sample rate in Hz
    :param duration: if given, truncate audio to this many seconds
    """
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    num_samples, num_channels = audio.shape

    if duration is not None:
        max_samples = int(duration * sample_rate)
        if num_samples > max_samples:
            audio = audio[:max_samples, :]
            num_samples = max_samples

    # Normalize to 16-bit PCM range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    audio_int16 = np.int16(audio * 32767)

    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def export_soundscape(
    engine,
    filepath: str,
    duration: Optional[float] = None,
    sample_rate: int = 44100,
) -> None:
    """
    Generate soundscape from engine and export as WAV.

    :param engine: SoundscapeEngine instance (must have generate_block method)
    :param filepath: output .wav file path
    :param duration: length of audio to generate (None = use engine default)
    :param sample_rate: sample rate in Hz
    """
    import time

    if duration is None:
        duration = 30.0  # default 30 seconds

    block_size = int(sample_rate * 0.1)  # 100ms blocks
    total_samples = int(duration * sample_rate)
    num_channels = 2  # stereo

    audio_buffer = np.zeros((total_samples, num_channels), dtype=np.float64)
    samples_written = 0

    while samples_written < total_samples:
        remaining = total_samples - samples_written
        this_block = min(block_size, remaining)
        block = engine.generate_block(this_block)
        if block is None:
            break
        if isinstance(block, np.ndarray) and block.ndim == 1:
            block = block[:, np.newaxis]
        if block.shape[1] < num_channels:
            # Mono to stereo
            block = np.tile(block, (1, num_channels))
        audio_buffer[samples_written:samples_written + this_block, :] = block[:this_block, :]
        samples_written += this_block

    export_to_wav(audio_buffer[:samples_written, :], filepath, sample_rate)
