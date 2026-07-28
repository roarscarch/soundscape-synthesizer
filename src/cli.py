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
        help="Biome preset to use (default: 'forest')",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duration in seconds (0 = infinite, default: 0)",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )

    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export the generated soundscape to a WAV file at the given path",
    )

    parser.add_argument(
        "--volume",
        type=float,
        default=0.5,
        help="Master volume (0.0 to 1.0, default: 0.5)",
    )

    parser.add_argument(
        "--sleep-timer",
        type=int,
        default=0,
        help="Sleep timer in minutes (0 = disabled, default: 0). After this many minutes, the soundscape will fade out and exit.",
    )

    parser.add_argument(
        "--fade-duration",
        type=int,
        default=30,
        help="Fade-out duration in seconds when sleep timer expires (default: 30)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    """Main entry point for the soundscape synthesizer."""
    args = parse_args(argv)

    # Validate arguments
    if args.volume < 0.0 or args.volume > 1.0:
        print("Error: --volume must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    if args.sleep_timer < 0:
        print("Error: --sleep-timer must be a non-negative integer", file=sys.stderr)
        sys.exit(1)

    if args.fade_duration < 0:
        print("Error: --fade-duration must be a non-negative integer", file=sys.stderr)
        sys.exit(1)

    # Resolve biome
    if args.biome not in BIOME_REGISTRY:
        print(f"Error: Unknown biome '{args.biome}'. Available biomes: {list(BIOME_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    biome = BIOME_REGISTRY[args.biome]

    # Create engine and player
    engine = SoundscapeEngine(
        biome=biome,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )

    player = AudioPlayer(
        engine=engine,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )

    # Handle graceful shutdown
    running = [True]

    def signal_handler(sig, frame):
        print("\nShutting down...")
        running[0] = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # If sleep timer is set, schedule fade-out
    fade_start_time = None
    if args.sleep_timer > 0:
        fade_start_time = time.time() + args.sleep_timer * 60
        print(f"Sleep timer set for {args.sleep_timer} minute(s). Fade-out duration: {args.fade_duration}s")

    # Start playback
    print(f"Starting soundscape: seed='{args.seed}', biome='{args.biome}', volume={args.volume}")
    if args.duration > 0:
        print(f"Duration: {args.duration}s")
    else:
        print("Duration: infinite (press Ctrl+C to stop)")

    player.start(volume=args.volume)

    try:
        start_time = time.time()
        while running[0]:
            elapsed = time.time() - start_time

            # Check duration limit
            if args.duration > 0 and elapsed >= args.duration:
                print(f"\nDuration limit reached ({args.duration}s).")
                break

            # Check sleep timer
            if fade_start_time is not None:
                remaining = fade_start_time - time.time()
                if remaining <= 0:
                    # Begin fade-out
                    print(f"\nSleep timer expired. Fading out over {args.fade_duration}s...")
                    player.fade_out(duration=args.fade_duration)
                    # Wait for fade-out to complete
                    time.sleep(args.fade_duration + 0.5)
                    break
                elif remaining <= 5.0:
                    # Warn user
                    print(f"\nSleep timer ending in {int(remaining)} seconds...")

            # Sleep a bit to avoid busy-waiting
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        player.stop()

    # Export if requested
    if args.export:
        print(f"Exporting soundscape to '{args.export}'...")
        try:
            player.export_wav(args.export)
            print("Export complete.")
        except Exception as e:
            print(f"Export failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
