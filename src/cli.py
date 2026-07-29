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
        default=0.0,
        help="Duration in seconds (0 = infinite, default: 0)",
    )

    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export soundscape to WAV file (e.g., output.wav)",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep timer in minutes (0 = disabled, default: 0)",
    )

    parser.add_argument(
        "--volume",
        type=float,
        default=0.5,
        help="Initial volume (0.0 to 1.0, default: 0.5)",
    )

    parser.add_argument(
        "--list-biomes",
        action="store_true",
        help="List available biomes and exit",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Display detailed info about the selected biome and exit",
    )

    args = parser.parse_args(argv)

    # Validate seed: must be non-empty string
    if not args.seed or not args.seed.strip():
        parser.error("Seed phrase cannot be empty.")

    # Validate volume range
    if args.volume < 0.0 or args.volume > 1.0:
        parser.error("Volume must be between 0.0 and 1.0.")

    # Validate duration
    if args.duration < 0:
        parser.error("Duration must be non-negative.")

    # Validate sleep
    if args.sleep < 0:
        parser.error("Sleep timer must be non-negative.")

    return args


def list_biomes() -> None:
    """Print available biomes."""
    print("Available biomes:")
    for name, biome in BIOME_REGISTRY.items():
        print(f"  {name}: {biome.name}")
        print(f"    base frequencies: {biome.base_frequencies}")
        print(f"    harmonics: {list(biome.harmonics)}")
        print(f"    grain duration: {biome.grain_duration}s")
        print()


def show_biome_info(biome_name: str) -> None:
    """Display detailed info about a specific biome."""
    biome = BIOME_REGISTRY.get(biome_name)
    if not biome:
        print(f"Error: Biome '{biome_name}' not found.")
        sys.exit(1)
    print(f"Biome: {biome.name}")
    print(f"  Base frequencies: {biome.base_frequencies}")
    print(f"  Harmonics: {list(biome.harmonics)}")
    print(f"  Envelope attack: {biome.envelope_attack}s")
    print(f"  Envelope decay: {biome.envelope_decay}s")
    print(f"  Grain duration: {biome.grain_duration}s")
    print(f"  Sample rate: {biome.sample_rate} Hz")
    print(f"  Seed: {biome.seed if biome.seed else 'None (default)'}")


def main() -> None:
    """Main entry point for the soundscape synthesizer."""
    args = parse_args()

    # Handle special flags
    if args.list_biomes:
        list_biomes()
        return

    if args.info:
        show_biome_info(args.biome)
        return

    # Get biome
    biome = BIOME_REGISTRY.get(args.biome)
    if not biome:
        print(f"Error: Biome '{args.biome}' not found.")
        sys.exit(1)

    print(f"Starting soundscape with seed: '{args.seed}'")
    print(f"Biome: {biome.name}")
    print(f"Volume: {args.volume}")
    if args.duration > 0:
        print(f"Duration: {args.duration}s")
    else:
        print("Duration: infinite")
    if args.sleep > 0:
        print(f"Sleep timer: {args.sleep} minutes")
    if args.export:
        print(f"Export to: {args.export}")
    print()

    # Initialize engine and player
    try:
        from .grain_bank import GrainBank
        grain_bank = GrainBank(
            biome=biome,
            seed=args.seed,
        )
        engine = SoundscapeEngine(
            seed=args.seed,
            grain_bank=grain_bank,
        )
        player = AudioPlayer(
            engine=engine,
            sample_rate=biome.sample_rate,
            volume=args.volume,
        )
    except Exception as e:
        print(f"Error initializing audio system: {e}")
        sys.exit(1)

    # Handle sleep timer
    sleep_seconds = args.sleep * 60.0
    if sleep_seconds > 0:
        fade_duration = 5.0  # 5 seconds fade-out
        player.set_sleep_timer(sleep_seconds, fade_duration)

    # Handle export
    if args.export:
        try:
            player.export_to_wav(args.export, duration=args.duration if args.duration > 0 else 60.0)
            print(f"Exported soundscape to {args.export}")
            return
        except Exception as e:
            print(f"Error exporting soundscape: {e}