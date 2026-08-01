"""Export soundscape to WAV file with configurable format and bit depth."""

import wave
import numpy as np
from typing import Optional, Tuple


def _validate_params(
    sample_rate: int, channels: int, format: str, bit_depth: int
) -> None:
    """Validate export parameters."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")
    if format not in ("wav", "wave"):
        raise ValueError(f"Unsupported format '{format}', only 'wav' is supported")
    if bit_depth not in (16, 24, 32):
        raise ValueError(f"bit_depth must be 16, 24, or 32, got {bit_depth}")
    if format == "wav" and bit_depth not in (16, 24):
        raise ValueError("WAV format only supports 16 or 24 bit depth")


def _to_pcm(
    audio: np.ndarray, bit_depth: int, dtype: np.dtype
) -> np.ndarray:
    """Convert float audio to integer PCM."""
    if bit_depth == 16:
        scale = 32767
    elif bit_depth == 24:
        scale = 8388607
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    # Clip to [-1.0, 1.0] and convert
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int = (audio_clipped * scale).astype(dtype)
    return audio_int


def export_wav(
    filepath: str,
    audio: np.ndarray,
    sample_rate: int,
    channels: int = 2,
    bit_depth: int = 16,
    format: str = "wav",
) -> None:
    """
    Export audio data to a WAV file.

    Args:
        filepath: Output file path (must end with .wav)
        audio: Audio data as float array in shape (samples, channels) or (samples,)
        sample_rate: Sample rate in Hz
        channels: Number of channels (1 or 2, default 2)
        bit_depth: Bit depth (16 or 24, default 16)
        format: Output format (only 'wav' supported)

    Raises:
        ValueError: If parameters are invalid or audio shape doesn't match channels
        OSError: If file cannot be written
    """
    # Validate parameters
    _validate_params(sample_rate, channels, format, bit_depth)

    # Convert audio to float32 if needed
    audio = np.asarray(audio, dtype=np.float32)

    # Ensure audio is 2D with correct channel count
    if audio.ndim == 1:
        if channels != 1:
            raise ValueError(
                f"Audio is 1D but channels={channels}, expected 2D array"
            )
        audio = audio.reshape(-1, 1)
    elif audio.ndim == 2:
        if audio.shape[1] != channels:
            raise ValueError(
                f"Audio has {audio.shape[1]} channels but channels={channels}"
            )
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape {audio.shape}")

    # Convert to PCM
    if bit_depth == 16:
        dtype = np.int16
    elif bit_depth == 24:
        dtype = np.int32  # stored as 3 bytes per sample
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    pcm = _to_pcm(audio, bit_depth, dtype)

    # Open WAV file
    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

        if bit_depth == 24:
            # Convert int32 to 3-byte little-endian format
            # Each sample: 4 bytes, we drop the top byte
            pcm_bytes = pcm.astype("<i4").tobytes()
            # Extract 3 bytes per sample (little-endian)
            sample_count = pcm.size
            out = bytearray(sample_count * 3)
            for i in range(sample_count):
                val = int(pcm.flat[i])
                # Clamp to 24-bit range
                val = max(-8388608, min(8388607, val))
                # Little-endian: bytes 0,1,2
                out[i*3] = (val & 0xFF)
                out[i*3+1] = ((val >> 8) & 0xFF)
                out[i*3+2] = ((val >> 16) & 0xFF)
            wav_file.writeframes(bytes(out))
        else:
            wav_file.writeframes(pcm.tobytes())


def export_mixdown(
    filepath: str,
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    bit_depth: int = 16,
) -> None:
    """
    Export a stereo mixdown to WAV file.

    Args:
        filepath: Output file path
        left: Left channel audio (float array)
        right: Right channel audio (float array)
        sample_rate: Sample rate
        bit_depth: Bit depth (16 or 24)
    """
    if left.shape != right.shape:
        raise ValueError(f"Left and right channels must have same shape, got {left.shape} and {right.shape}