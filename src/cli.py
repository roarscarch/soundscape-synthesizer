"""
Command-line interface for Soundscape Synthesizer.
Provides argument parsing and main entry point.
"""

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
        help="Biome preset (default: forest)",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Playback duration in seconds (infinite if not set)",
    )

    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export soundscape to WAV file (overrides playback)",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of audio channels: 1 mono, 2 stereo (default: 2)",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="Size of the wave function collapse grid (default: 8)",
    )

    parser.add_argument(
        "--volume",
        type=float,
        default=0.7,
        help="Initial volume 0.0 to 1.0 (default: 0.7)",
    )

    parser.add_argument(
        "--fade-out",
        type=float,
        default=5.0,
        help="Fade-out duration in seconds (default: 5.0)",
    )

    parser.add_argument(
        "--list-biomes",
        action="store_true",
        help="List available biomes and exit",
    )

    args = parser.parse_args(argv)
    return args


def list_biomes() -> None:
    """Print available biomes and their descriptions."""
    print("Available biomes:")
    for name, biome_cls in BIOME_REGISTRY.items():
        print(f"  {name}: {biome_cls.description}")
    print()


def run(args: argparse.Namespace) -> None:
    """Main execution: export or play the soundscape."""
    if args.list_biomes:
        list_biomes()
        return

    # Get biome class
    biome_cls = BIOME_REGISTRY.get(args.biome)
    if biome_cls is None:
        print(f"Error: Unknown biome '{args.biome}'. Use --list-biomes to see available options.")
        sys.exit(1)

    # Instantiate biome
    biome = biome_cls(seed=args.seed, sample_rate=args.sample_rate)

    # Create engine
    engine = SoundscapeEngine(
        biome=biome,
        grid_size=args.grid_size,
        sample_rate=args.sample_rate,
    )

    # Create audio player
    player = AudioPlayer(
        sample_rate=args.sample_rate,
        channels=args.channels,
    )

    # Set grain callback
    player.set_grain_callback(engine.generate_grain)

    # Set initial volume
    player.set_volume(args.volume)

    # Handle export mode
    if args.export:
        print(f"Exporting soundscape to {args.export}...")
        duration = args.duration if args.duration is not None else 60.0
        player.export_wav(args.export, duration)
        print("Export complete.")
        return

    # Playback mode
    print(f"Starting soundscape: seed='{args.seed}', biome='{args.biome}'")
    print("Press Ctrl+C to stop.")

    player.start()

    # Set up signal handler for graceful shutdown
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        print("\nStopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.duration is not None:
        stop_event.wait(args.duration)
        print("Duration reached. Fading out...")
        player.fade_out(args.fade_out)
        time.sleep(args.fade_out + 0.5)
    else:
        stop_event.wait()

    player.stop()
    print("Soundscape stopped.")


def main(argv: Optional[list] = None) -> None:
    """Entry point for the CLI."""
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
