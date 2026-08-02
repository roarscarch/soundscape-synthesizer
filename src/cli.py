import argparse
import sys
import signal
import time
from typing import Optional

from .biomes import Biome, BIOME_REGISTRY
from .engine import SoundscapeEngine
from .player import AudioPlayer


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate infinite, evolving ambient soundscapes from a seed phrase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli --seed "forest morning" --biome forest
  python -m src.cli --seed "deep space" --biome space --duration 300
  python -m src.cli --seed "ocean waves" --biome ocean --export soundscape.wav
        """,
    )

    parser.add_argument(
        "--seed",
        type=str,
        default="soundscape",
        help="Seed phrase for deterministic generation (default: 'soundscape')",
    )

    parser.add_argument(
        "--biome",
        type=str,
        choices=list(BIOME_REGISTRY.keys()),
        default="forest",
        help=f"Biome preset (choices: {', '.join(BIOME_REGISTRY.keys())}, default: forest)",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (default: infinite)",
    )

    parser.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="FILE",
        help="Export the generated soundscape to a WAV file and exit",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["wav", "flac", "ogg"],
        default="wav",
        help="Export format (default: wav)",
    )

    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=[16, 24, 32],
        default=16,
        help="Bit depth for export (default: 16)",
    )

    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Master volume (0.0 to 1.0, default: 1.0)",
    )

    parser.add_argument(
        "--pan-lfo-rate",
        type=float,
        default=None,
        help="Pan LFO rate in Hz (overrides biome default)",
    )

    parser.add_argument(
        "--volume-lfo-rate",
        type=float,
        default=None,
        help="Volume LFO rate in Hz (overrides biome default)",
    )

    parser.add_argument(
        "--stereo-width",
        type=float,
        default=1.0,
        help="Stereo width factor (0.0 = mono, 1.0 = full stereo, default: 1.0)",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Sleep timer in seconds before auto fade-out and exit",
    )

    parser.add_argument(
        "--fade",
        type=float,
        default=5.0,
        help="Fade-out duration in seconds (default: 5.0)",
    )

    parser.add_argument(
        "--no-meter",
        action="store_true",
        help="Disable real-time audio level meter",
    )

    return parser.parse_args(argv)


def get_biome_from_seed(seed: str, biome_name: str) -> Biome:
    """
    Get a biome instance, optionally varying its parameters based on the seed.
    This ensures that the same seed always produces the same biome, but different
    seeds produce slightly different variations of the same biome.
    """
    import hashlib

    base = BIOME_REGISTRY[biome_name]
    # Create a deterministic hash from the seed phrase to vary parameters.
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    # Use the first 4 bytes as a float in [0, 1)
    variation = int.from_bytes(digest[:4], "big") / (2**32 - 1)

    # Vary base frequencies by up to +/- 5% (but keep relative spacing)
    freq_scale = 1.0 + (variation - 0.5) * 0.1
    base_frequencies = [f * freq_scale for f in base.base_frequencies]

    # Vary envelope times slightly (e.g., up to +/- 20%)
    attack_scale = 1.0 + (variation - 0.5) * 0.4
    decay_scale = 1.0 + ((variation * 7) % 1.0 - 0.5) * 0.4

    # Vary grain duration slightly
    grain_duration = base.grain_duration * (1.0 + (variation - 0.5) * 0.2)

    # Create a new Biome with the varied parameters.
    return Biome(
        name=base.name,
        base_frequencies=base_frequencies,
        harmonics=base.harmonics,
        envelope_attack=base.envelope_attack * attack_scale,
        envelope_decay=base.envelope_decay * decay_scale,
        grain_duration=grain_duration,
        sample_rate=base.sample_rate,
        seed=seed,  # Pass the seed to ensure deterministic wave table
        pan_spread=base.pan_spread,
        volume_lfo_rate=base.volume_lfo_rate,
        pan_lfo_rate=base.pan_lfo_rate,
    )


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the CLI."""
    args = parse_args(argv)

    # Build the biome with seed-based variation
    biome = get_biome_from_seed(args.seed, args.biome)

    # Override LFO rates if specified
    if args.volume_lfo_rate is not None:
        biome.volume_lfo_rate = args.volume_lfo_rate
    if args.pan_lfo_rate is not None:
        biome.pan_lfo_rate = args.pan_lfo_rate

    # Build the engine
    engine = SoundscapeEngine(
        biome=biome,
        seed_phrase=args.seed,
        sample_rate=44100,
    )

    # Set stereo width
    engine.stereo_width = args.stereo_width

    # Set master volume
    engine.master_volume = args.volume

    # Configure fade-out
    if args.fade > 0:
        engine.fade_duration = args.fade

    # Configure sleep timer
    if args.sleep is not None and args.sleep > 0:
        engine.sleep_timer = args.sleep

    # Set up export if requested
    if args.export:
        engine.export_path = args.export
        engine.export_format = args.format
        engine.export_bit_depth = args.bit_depth

    # Create audio player
    player = AudioPlayer(
        engine=engine,
        sample_rate=44100,
        channels=2,
        enable_meter=not args.no_meter,
    )

    # Install signal handlers for graceful exit
    def handle_signal(signum, frame):
        print("\
Stopping...")
        player.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        player.start()
        # If a duration is specified, stop after that duration
        if args.duration is not None:
            print(f"Playing for {args.duration}