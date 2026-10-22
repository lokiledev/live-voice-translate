# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Terminal shortcuts. They only arrive while the terminal is focused."""

from __future__ import annotations

import select
import sys
import termios
import tty
from datetime import datetime
from typing import TYPE_CHECKING

from live_voice_translate.languages import SOURCE_FLAG, SOURCE_LABEL, TARGET_LABEL
from live_voice_translate.models import MODES, WHISPER_MODELS, ModelConfig
from live_voice_translate.native import WhisperModel
from live_voice_translate.transcript import TranscriptWriter

if TYPE_CHECKING:
    from live_voice_translate.translator import LiveTranslator


class KeyboardController:
    """Read single keypresses with ``select`` and apply them to the session."""

    def __init__(self, translator: LiveTranslator) -> None:
        """Attach to a running translator."""
        self.translator = translator
        self.old_settings = None
        self.shortcuts = {
            "p": "Pause/Resume",
            "s": "Save transcript now",
            "m": "Change mode (fast/normal/slow)",
            "w": "Change Whisper model (tiny/base/small/medium/turbo/large)",
            "i": "Toggle Chinese display",
            "q": "Quit",
            "h": "Show this help",
        }

    def setup(self) -> bool:
        """Put the terminal in cbreak mode so keys arrive without Enter."""
        if not sys.stdin.isatty():
            return False
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print("\n💡 Keyboard shortcuts enabled:")
            self._show_help()
        except Exception as exc:
            print(f"\n⚠️  Could not enable keyboard shortcuts: {exc}")
            return False
        return True

    def cleanup(self) -> None:
        """Restore the terminal."""
        if self.old_settings is None:
            return
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        except Exception:
            return

    def get_key(self, timeout: float = 0.01) -> str | None:
        """Return one printable character, or None if nothing was pressed."""
        try:
            if select.select([sys.stdin], [], [], timeout)[0]:
                char = sys.stdin.read(1)
                if char and len(char) == 1 and char.isprintable():
                    return char
        except Exception:
            return None
        return None

    def handle_key(self, key: str | None) -> None:
        """Apply one shortcut. Other keys are ignored."""
        if not key or len(key) != 1:
            return
        key = key.lower()
        if key == "p":
            self._toggle_pause()
        elif key == "s":
            self._save_now()
        elif key == "m":
            self._change_mode()
        elif key == "w":
            self._change_model()
        elif key == "i":
            self._toggle_chinese()
        elif key == "q":
            self._quit()
        elif key == "h":
            self._show_help()

    def _toggle_pause(self) -> None:
        self.translator.paused = not self.translator.paused
        if self.translator.paused:
            print("\n⏸️  PAUSED. Press 'p' to resume.", flush=True)
        else:
            print("\n▶️  RESUMED", flush=True)

    def _save_now(self) -> None:
        """Flush the transcript, creating a file first when none was requested."""
        if not self.translator.writer.filepath:
            filename = f"live-translate-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
            print(f"\n💾 Creating save file: {filename}", flush=True)
            self.translator.writer = TranscriptWriter(
                filename,
                self.translator.model_name,
                self.translator.mode,
            )
            if self.translator.writer.filepath:
                print(f"✅ Now saving to: {filename}", flush=True)
            else:
                print("❌ Could not create save file", flush=True)
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n💾 [{timestamp}] Saving transcript...", flush=True)
        if self.translator.writer.file_handle:
            self.translator.writer.file_handle.flush()
        print(f"✅ Saved to: {self.translator.writer.filepath}", flush=True)

    def _change_mode(self) -> None:
        """Cycle fast, then normal, then slow."""
        current_idx = MODES.index(self.translator.mode)
        new_mode = MODES[(current_idx + 1) % len(MODES)]
        self.translator.mode = new_mode
        self.translator.config = ModelConfig.get_config(self.translator.model_name, new_mode)
        desc = ModelConfig.MODE_DESC.get(new_mode, new_mode)
        print(f"\n🔄 Mode changed: {new_mode.upper()} ({desc}) — effective immediately", flush=True)

    def _change_model(self) -> None:
        """Cycle Whisper models. Audio stays paused while the weights load."""
        current_idx = WHISPER_MODELS.index(self.translator.model_name)
        new_model = WHISPER_MODELS[(current_idx + 1) % len(WHISPER_MODELS)]
        print(f"\n🔁 Switching to model {new_model} — pausing audio...", flush=True)

        was_paused = self.translator.paused
        self.translator.paused = True
        try:
            print(f"⏳ Loading {new_model}...", flush=True)
            device = "cuda" if self.translator.use_gpu else "cpu"
            compute_type = "float16" if self.translator.use_gpu else "int8"
            self.translator.model = WhisperModel(
                new_model, device=device, compute_type=compute_type
            )
            self.translator.model_name = new_model
            self.translator.config = ModelConfig.get_config(new_model, self.translator.mode)
            print(f"✅ Model: {new_model} (beam={self.translator.config['beam']})", flush=True)
        except Exception as exc:
            print(f"⚠️  Failed to load {new_model}: {exc}", flush=True)
        finally:
            self.translator.paused = was_paused

    def _toggle_chinese(self) -> None:
        """Toggle the Chinese source text in the terminal."""
        self.translator.show_chinese = not self.translator.show_chinese
        state = "ON" if self.translator.show_chinese else "OFF"
        print(f"\n{SOURCE_FLAG} Chinese display: {state}", flush=True)

    def _quit(self) -> None:
        print("\n\n👋 Quitting...", flush=True)
        self.translator.should_quit = True

    def _show_help(self) -> None:
        session = self.translator
        print("\n  Current session")
        print("  " + "-" * 34)
        print(f"  Model    : {session.model_name}")
        print(f"  Mode     : {session.mode}")
        print(f"  Language : {SOURCE_LABEL} -> {TARGET_LABEL}")
        print(f"  VAD      : {'on' if session.vad_filter else 'off'}")
        print(f"  Phrases  : {session.phrase_count}  Words: {session.word_count}")
        print()
        print("  Keyboard Shortcuts")
        print("  " + "-" * 34)
        for key, description in self.shortcuts.items():
            print(f"  {key.upper()}  ->  {description}")
        print()
        print("  Note: shortcuts are processed between audio chunks.")
        print("  Response may be delayed up to one chunk duration.")
        print()
