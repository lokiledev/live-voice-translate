# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line entry point."""

import argparse
import sys
from datetime import datetime

from live_voice_translate.audio import AudioCapture
from live_voice_translate.cuda import cuda_available, cuda_device_name
from live_voice_translate.languages import SOURCE_LABEL, TARGET_LABEL
from live_voice_translate.models import MODEL_ALIASES, ModelConfig
from live_voice_translate.translator import LiveTranslator


def show_menu() -> str:
    """Ask which Whisper model to load. Enter selects large-v3-turbo."""
    print("═══════════════════════════════════════════")
    print("   Chinese → English Live Translation")
    print("═══════════════════════════════════════════")
    print()
    print("Available models:")
    print()
    print("  1) tiny   - Ultra fast     (60% accuracy, ~1.5s, 1GB RAM)")
    print("  2) base   - Fast           (85% accuracy, ~4s, 1.5GB RAM)")
    print("  3) small  - Balanced       (90% accuracy, ~5s, 2GB RAM)")
    print("  4) medium - Balanced       (high accuracy, ~5GB RAM)")
    print("  5) turbo  - Recommended    (high accuracy, lower latency) [DEFAULT]")
    print("  6) large  - Maximum        (best accuracy, ~10GB RAM) ⚠️  High fan")
    print()

    choice = input("Your choice (1-6 or Enter for default): ").strip()
    models = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large-v3-turbo",
        "6": "large-v3",
        "": "large-v3-turbo",
    }
    return models.get(choice, "large-v3-turbo")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time Chinese→English audio translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive menu
  %(prog)s medium                    # Medium model
  %(prog)s medium --save             # Save to auto-generated file
  %(prog)s large --slow --save meeting.md
  %(prog)s turbo --show-chinese      # Display Chinese + English
  %(prog)s turbo --mixed             # Mixed Chinese/English speech
  %(prog)s turbo --overlay           # Floating window: last sentence + translation
  %(prog)s medium --no-vad           # Disable Voice Activity Detection

Audio source:
  If multiple audio streams are active (e.g. Zoom + YouTube), an
  interactive menu lets you pick which one to translate.

Keyboard shortcuts (during execution):
  p - Pause/Resume
  s - Save transcript now (creates file if needed)
  m - Change mode (fast/normal/slow)
  w - Change Whisper model (tiny/base/small/medium/turbo/large)
  i - Toggle Chinese display
  q - Quit
  h - Show help
        """,
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=["tiny", "base", "small", "medium", "turbo", "large"],
        help="Whisper model to use",
    )
    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help="500ms silence threshold, reduced beam — reactive, may cut mid-sentence",
    )
    parser.add_argument(
        "--slow",
        "-s",
        action="store_true",
        help="1200ms silence threshold, max beam — waits for complete sentences",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=f"live-translate-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md",
        metavar="FILE",
        help="Save transcript to file (default: auto-generated)",
    )
    parser.add_argument(
        "--show-chinese",
        "--show-italian",
        "-i",
        dest="show_chinese",
        action="store_true",
        help="Display Chinese source text (toggle with 'i' key)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Transcribe on the NVIDIA GPU (this is the default when CUDA is available)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU transcription",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Audio mixes Chinese and English: translate Chinese parts, show English parts as is",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Floating half-transparent window with the last sentence and its translation",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Whisper's internal VAD filter (webrtcvad chunking is always active)",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable keyboard shortcuts",
    )
    return parser


def _select_stream() -> tuple[str, str]:
    """Return the PipeWire node name to monitor and a description of it."""
    streams = AudioCapture.get_active_streams()
    if not streams:
        default_sink = AudioCapture.get_default_sink()
        if default_sink is None:
            print("Error: No PipeWire audio output device found", file=sys.stderr)
            sys.exit(1)
        _sink_id, stream_id, stream_desc = default_sink
        return stream_id, stream_desc + " (no audio playing yet, waiting)"
    if len(streams) == 1:
        _sink_id, stream_id, stream_desc = streams[0]
        return stream_id, stream_desc

    print("  Multiple audio streams detected:")
    print()
    for index, (_sink_id, _name, desc) in enumerate(streams, 1):
        print(f"    {index}) {desc}")
    print()
    choice = input(f"  Select stream (1-{len(streams)}): ").strip()
    try:
        selected = int(choice) - 1
        if not 0 <= selected < len(streams):
            raise ValueError
    except ValueError:
        print("Invalid choice, using first stream.")
        selected = 0
    _sink_id, stream_id, stream_desc = streams[selected]
    return stream_id, stream_desc


def main() -> None:
    """Parse arguments, pick an output device, and run until quit."""
    args = _build_parser().parse_args()

    if args.cpu and args.gpu:
        print("Choose either --gpu or --cpu.", file=sys.stderr)
        sys.exit(2)
    use_gpu = False if args.cpu else args.gpu or cuda_available()

    if args.model:
        model = MODEL_ALIASES.get(args.model, args.model)
    else:
        model = show_menu()

    if args.fast:
        mode = "fast"
    elif args.slow:
        mode = "slow"
    else:
        mode = "normal"

    config = ModelConfig.get_config(model, mode)

    print()
    print("═══════════════════════════════════════════")
    print("   Configuration")
    print("═══════════════════════════════════════════")
    print()
    print(f"  Model      : {model}")
    print(f"  Accuracy   : {config['precision']}")
    print(f"  Mode       : {mode.capitalize()} — {ModelConfig.MODE_DESC[mode]}")
    if args.save:
        print(f"  Save to    : {args.save}")
    if args.show_chinese:
        print("  Chinese    : Displayed")
    source = f"{SOURCE_LABEL} + {TARGET_LABEL}" if args.mixed else SOURCE_LABEL
    print(f"  Language   : {source} → {TARGET_LABEL}")
    print(f"  Device     : {cuda_device_name() if use_gpu else 'CPU'}")
    print(f"  VAD        : {'Disabled' if args.no_vad else 'Enabled (skips silence)'}")
    if args.overlay:
        print("  Overlay    : Last sentence (source + translation)")
    print()

    stream_id, stream_desc = _select_stream()
    print(f"  Stream     : {stream_desc}")
    print()
    print("═══════════════════════════════════════════")
    print()
    print("Translation in progress... (Ctrl+C to stop)")
    print()

    translator = LiveTranslator(
        model,
        mode,
        args.save,
        enable_keyboard=not args.no_keyboard,
        show_chinese=args.show_chinese,
        mixed=args.mixed,
        vad_filter=not args.no_vad,
        use_gpu=use_gpu,
        overlay=args.overlay,
    )
    translator.setup()
    translator.run(stream_id)
