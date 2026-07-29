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
        help="Duration in seconds for the soundscape (default: infinite)",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Sleep timer in minutes after which the soundscape fades out and stops",
    )

    parser.add_argument(
        "--fade",
        type=float,
        default=10.0,
        help="Fade-out duration in seconds when sleep timer expires (default: 10.0)",
    )

    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Path to export the current soundscape as a WAV file (ongoing export when used with --duration)",
    )

    parser.add_argument(
        "--no-visualizer",
        action="store_true",
        help="Disable the real-time visualizer",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    """Entry point for the soundscape synthesizer."""
    args = parse_args(argv)

    # Validate seed
    if not args.seed.strip():
        print("Error: seed phrase cannot be empty.")
        sys.exit(1)

    # Load biome
    try:
        biome = BIOME_REGISTRY[args.biome]
    except KeyError:
        print(f"Error: unknown biome '{args.biome}'. Available: {', '.join(BIOME_REGISTRY.keys())}")
        sys.exit(1)

    print(f"Using biome: {biome.name}")
    print(f"Seed phrase: {args.seed}")

    # Initialize engine and player
    engine = SoundscapeEngine(seed=args.seed, grain_bank=biome)
    player = AudioPlayer(sample_rate=44100)

    # Handle export mode
    export_path = args.export

    # Set up sleep timer
    sleep_timer = args.sleep
    fade_duration = args.fade

    # Duration for generation (None means infinite)
    duration = args.duration

    # Graceful shutdown
    stop_event = [False]

    def signal_handler(sig, frame):
        print("\nShutting down...")
        stop_event[0] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start playback
    player.play(engine, duration=duration, sleep_timer=sleep_timer, fade_duration=fade_duration)

    # If export requested, wait for playback to finish then export
    if export_path:
        # For export, we need to run for the full duration or until interrupted
        if duration is not None:
            # Run until duration expires
            start_time = time.time()
            while time.time() - start_time < duration:
                if stop_event[0]:
                    break
                time.sleep(0.1)
            player.stop()
        elif sleep_timer is not None:
            # Run until sleep timer triggers (handled by player)
            while not stop_event[0]:
                time.sleep(0.1)
        else:
            # Infinite mode with export: user must interrupt
            print("Exporting in infinite mode. Press Ctrl+C to stop and export.")
            while not stop_event[0]:
                time.sleep(0.1)

        # Export the audio buffer
        try:
            player.export(export_path)
            print(f"Exported soundscape to {export_path}")
        except Exception as e:
            print(f"Export failed: {e}")
            sys.exit(1)
    else:
        # No export: block until playback ends
        while player.is_playing() and not stop_event[0]:
            time.sleep(0.1)
        player.stop()


if __name__ == "__main__":
    main()
