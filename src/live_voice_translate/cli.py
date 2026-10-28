# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line entry point."""

import sys
from datetime import datetime
from typing import Annotated, Literal

from cyclopts import App, Parameter

from live_voice_translate.audio import AudioCapture
from live_voice_translate.cuda import cuda_available, cuda_device_name
from live_voice_translate.languages import SOURCE_LABEL, TARGET_LABEL
from live_voice_translate.models import MODEL_ALIASES, ModelConfig
from live_voice_translate.translator import LiveTranslator

ModelChoice = Literal["tiny", "base", "small", "medium", "turbo", "large"]

app = App(
    name="lvt",
    help="Real-time Chinese→English audio translation",
    help_format="plaintext",
    version_flags=[],
    help_epilogue="""
Examples:
  lvt                                # turbo (large-v3-turbo) by default
  lvt medium                         # Medium model
  lvt medium --save                  # Save to auto-generated file
  lvt large --slow --save meeting.md
  lvt turbo --show-chinese           # Display Chinese + English
  lvt turbo --mixed                  # Mixed Chinese/English speech
  lvt --overlay                      # turbo, with the floating window
  lvt medium --no-vad                # Disable Voice Activity Detection

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


def _flag(
    help_text: str,
    *,
    alias: str | None = None,
    name: str | list[str] | None = None,
    metavar: str | None = None,
) -> Parameter:
    """A flag without an automatic --no- form."""
    return Parameter(
        help=help_text,
        negative="",
        alias=alias,
        name=name,
        metavar=metavar,
        show_default=False,
    )


@app.default
def translate(
    model: Annotated[
        ModelChoice,
        Parameter(help="Whisper model to use"),
    ] = "turbo",
    *,
    fast: Annotated[
        bool,
        _flag("500ms silence threshold, reduced beam — reactive, may cut mid-sentence", alias="-f"),
    ] = False,
    slow: Annotated[
        bool,
        _flag("1200ms silence threshold, max beam — waits for complete sentences", alias="-s"),
    ] = False,
    save: Annotated[
        bool | str,
        _flag(
            "Save transcript to file. Omit the path to auto-generate a name.",
            name="--save",
            metavar="[FILE]",
        ),
    ] = False,
    show_chinese: Annotated[
        bool,
        _flag(
            "Display Chinese source text (toggle with 'i' key)",
            name=["--show-chinese", "--show-italian", "-i"],
        ),
    ] = False,
    gpu: Annotated[
        bool,
        _flag("Transcribe on the NVIDIA GPU (this is the default when CUDA is available)"),
    ] = False,
    cpu: Annotated[bool, _flag("Force CPU transcription")] = False,
    mixed: Annotated[
        bool,
        _flag(
            "Audio mixes Chinese and English: translate Chinese parts, show English parts as is",
        ),
    ] = False,
    overlay: Annotated[
        bool,
        _flag("Floating half-transparent window with the last sentence and its translation"),
    ] = False,
    no_vad: Annotated[
        bool,
        _flag("Disable Whisper's internal VAD filter (Silero chunking is always active)"),
    ] = False,
    no_keyboard: Annotated[bool, _flag("Disable keyboard shortcuts")] = False,
) -> None:
    """Translate live audio until quit."""
    if cpu and gpu:
        print("Choose either --gpu or --cpu.", file=sys.stderr)
        sys.exit(2)
    use_gpu = False if cpu else gpu or cuda_available()

    resolved_model = MODEL_ALIASES.get(model, model)
    if fast:
        mode = "fast"
    elif slow:
        mode = "slow"
    else:
        mode = "normal"

    if save is True:
        save_path: str | None = f"live-translate-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    elif isinstance(save, str):
        save_path = save
    else:
        save_path = None

    config = ModelConfig.get_config(resolved_model, mode)

    print()
    print("═══════════════════════════════════════════")
    print("   Configuration")
    print("═══════════════════════════════════════════")
    print()
    print(f"  Model      : {resolved_model}")
    print(f"  Accuracy   : {config['precision']}")
    print(f"  Mode       : {mode.capitalize()} — {ModelConfig.MODE_DESC[mode]}")
    if save_path:
        print(f"  Save to    : {save_path}")
    if show_chinese:
        print("  Chinese    : Displayed")
    source = f"{SOURCE_LABEL} + {TARGET_LABEL}" if mixed else SOURCE_LABEL
    print(f"  Language   : {source} → {TARGET_LABEL}")
    print(f"  Device     : {cuda_device_name() if use_gpu else 'CPU'}")
    print(f"  VAD        : {'Disabled' if no_vad else 'Enabled (skips silence)'}")
    if overlay:
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
        resolved_model,
        mode,
        save_path,
        enable_keyboard=not no_keyboard,
        show_chinese=show_chinese,
        mixed=mixed,
        vad_filter=not no_vad,
        use_gpu=use_gpu,
        overlay=overlay,
    )
    translator.setup()
    translator.run(stream_id)


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
    app()
