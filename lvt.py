#!/usr/bin/env python3
"""
live-voice-translate - Real-time Chinese→English audio translation
Copyright (C) 2026 Stéphane de Labrusse

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author: Stéphane de Labrusse
"""

import argparse
import collections
import json
import os
import queue
from pathlib import Path
import re
import select
import subprocess
import sys
import termios
import textwrap
import time
import tty
from datetime import datetime
from threading import Thread

# Force UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# webrtcvad depends on pkg_resources which was removed in setuptools>=81
# Inject a minimal mock if it's missing so the import succeeds
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import types as _types
    _mock = _types.ModuleType("pkg_resources")
    _mock.get_distribution = lambda *a, **kw: _types.SimpleNamespace(version="0")  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = _mock

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import webrtcvad

def _preload_cuda12_cublas():
    """Make the pip-installed CUDA 12 cuBLAS visible to CTranslate2."""
    import ctypes
    import glob
    try:
        import nvidia.cublas
    except ImportError:
        return
    for path in nvidia.cublas.__path__:
        for lib in ("libcublasLt.so.12", "libcublas.so.12"):
            for match in glob.glob(f"{path}/lib/{lib}"):
                ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)


_preload_cuda12_cublas()

import argostranslate.package
import argostranslate.translate
import numpy as np
from faster_whisper import WhisperModel

# ============================================================================
# LANGUAGE CONSTANTS - single source of truth
# ============================================================================

SOURCE_LANG = "zh"
SOURCE_LABEL = "Chinese"
SOURCE_FLAG = "🇨🇳"
TARGET_LANG = "en"
TARGET_LABEL = "English"
TARGET_FLAG = "🇬🇧"


# ============================================================================
# KEYBOARD CONTROLLER (using select - no dependencies)
# ============================================================================

class KeyboardController:
    """Handle keyboard shortcuts using select (non-blocking, no dependencies)"""
    
    def __init__(self, translator):
        """
        Initialize keyboard controller
        
        Args:
            translator: LiveTranslator instance to control
        """
        self.translator = translator
        self.old_settings = None
        
        # Keyboard shortcuts help
        self.shortcuts = {
            'p': 'Pause/Resume',
            's': 'Save transcript now',
            'm': 'Change mode (fast/normal/slow)',
            'w': 'Change Whisper model (tiny/base/small/medium/turbo/large)',
            'i': 'Toggle Chinese display',
            'q': 'Quit',
            'h': 'Show this help',
        }
    
    def setup(self):
        """Setup terminal for non-blocking input"""
        if sys.stdin.isatty():
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                
                print("\n💡 Keyboard shortcuts enabled:")
                self._show_help()
                return True
            except Exception as e:
                print(f"\n⚠️  Could not enable keyboard shortcuts: {e}")
                return False
        return False
    
    def cleanup(self):
        """Restore terminal settings"""
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass
    
    def get_key(self, timeout=0.01):
        """
        Get key if pressed (non-blocking)
        
        Args:
            timeout: Timeout in seconds (default 0.01)
        
        Returns:
            str: Key pressed or None if no key
        """
        try:
            if select.select([sys.stdin], [], [], timeout)[0]:
                char = sys.stdin.read(1)
                # Only return single printable ASCII characters
                if char and len(char) == 1 and char.isprintable():
                    return char
            return None
        except Exception:
            return None
    
    def handle_key(self, key):
        """Handle keyboard shortcuts"""
        if not key:
            return
        
        # Convert to lowercase
        key = key.lower()
        
        # Only process single character keys
        if len(key) != 1:
            return
        
        # Handle shortcuts - STRICT checking
        if key == 'p':
            self._toggle_pause()
        elif key == 's':
            self._save_now()
        elif key == 'm':
            self._change_mode()
        elif key == 'w':
            self._change_model()
        elif key == 'i':
            self._toggle_chinese()
        elif key == 'q':
            self._quit()
        elif key == 'h':
            self._show_help()
        # Silently ignore other keys
    
    def _toggle_pause(self):
        """Toggle pause/resume"""
        self.translator.paused = not self.translator.paused
        
        if self.translator.paused:
            print("\n⏸️  PAUSED. Press 'p' to resume.", flush=True)
        else:
            print("\n▶️  RESUMED", flush=True)
    
    def _save_now(self):
        """Force save transcript now (create file if needed)"""
        # Si pas de fichier configuré, en créer un automatiquement
        if not self.translator.writer.filepath:
            # Générer nom de fichier auto
            filename = f"live-translate-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
            
            print(f"\n💾 Creating save file: {filename}", flush=True)
            
            # Créer le writer avec le nouveau fichier
            self.translator.writer = TranscriptWriter(
                filename,
                self.translator.model_name,
                self.translator.mode,
            )
            
            if self.translator.writer.filepath:
                print(f"✅ Now saving to: {filename}", flush=True)
            else:
                print(f"❌ Could not create save file", flush=True)
        else:
            # Fichier existe déjà, juste sauvegarder
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"\n💾 [{timestamp}] Saving transcript...", flush=True)
            if self.translator.writer.file_handle:
                self.translator.writer.file_handle.flush()
            print(f"✅ Saved to: {self.translator.writer.filepath}", flush=True)
    
    def _change_mode(self):
        """Cycle through modes: fast → normal → slow → fast"""
        modes = ['fast', 'normal', 'slow']
        current_idx = modes.index(self.translator.mode)
        new_idx = (current_idx + 1) % len(modes)
        new_mode = modes[new_idx]
        
        # Update mode and config
        self.translator.mode = new_mode
        self.translator.config = ModelConfig.get_config(
            self.translator.model_name,
            new_mode
        )
        
        # Show change
        desc = ModelConfig.MODE_DESC.get(new_mode, new_mode)
        print(f"\n🔄 Mode changed: {new_mode.upper()} ({desc}) — effective immediately", flush=True)
    
    def _change_model(self):
        """Cycle through Whisper models."""
        models = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
        current_idx = models.index(self.translator.model_name)
        new_model = models[(current_idx + 1) % len(models)]

        print(f"\n🔁 Switching to model {new_model} — pausing audio...", flush=True)

        # Pause to avoid filling the queue during the long load
        was_paused = self.translator.paused
        self.translator.paused = True

        try:
            print(f"⏳ Loading {new_model}...", flush=True)
            device = "cuda" if self.translator.use_gpu else "cpu"
            compute_type = "float16" if self.translator.use_gpu else "int8"
            self.translator.model = WhisperModel(new_model, device=device, compute_type=compute_type)
            self.translator.model_name = new_model
            self.translator.config = ModelConfig.get_config(new_model, self.translator.mode)
            print(f"✅ Model: {new_model} (beam={self.translator.config['beam']})", flush=True)
        except Exception as e:
            print(f"⚠️  Failed to load {new_model}: {e}", flush=True)
        finally:
            self.translator.paused = was_paused

    def _toggle_chinese(self):
        """Toggle Chinese source text display."""
        self.translator.show_chinese = not self.translator.show_chinese
        
        if self.translator.show_chinese:
            print(f"\n{SOURCE_FLAG} Chinese display: ON", flush=True)
        else:
            print(f"\n{SOURCE_FLAG} Chinese display: OFF", flush=True)
    
    def _quit(self):
        """Quit gracefully"""
        print("\n\n👋 Quitting...", flush=True)
        self.translator.should_quit = True
    
    def _show_help(self):
        """Show keyboard shortcuts and current session config"""
        t = self.translator
        print("\n  Current session")
        print("  " + "-" * 34)
        print(f"  Model    : {t.model_name}")
        print(f"  Mode     : {t.mode}")
        print(f"  Language : {SOURCE_LABEL} -> {TARGET_LABEL}")
        print(f"  VAD      : {'on' if t.vad_filter else 'off'}")
        print(f"  Phrases  : {t.phrase_count}  Words: {t.word_count}")
        print()
        print("  Keyboard Shortcuts")
        print("  " + "-" * 34)
        for key, description in self.shortcuts.items():
            print(f"  {key.upper()}  ->  {description}")
        print()
        print("  Note: shortcuts are processed between audio chunks.")
        print("  Response may be delayed up to one chunk duration.")
        print()


# ============================================================================
# CORE APPLICATION CODE
# ============================================================================

class AudioCapture:
    """Capture playback from PipeWire by monitoring the output device."""

    @staticmethod
    def _record_command(target):
        """Record a sink's monitor as 16 kHz mono PCM on stdout."""
        return [
            "pw-record",
            "--target", target,
            "--properties", "stream.capture.sink=true",
            "--rate", "16000",
            "--channels", "1",
            "--format", "s16",
            "--raw",
            "-",
        ]

    @staticmethod
    def _pw_dump():
        try:
            result = subprocess.run(
                ["pw-dump"],
                capture_output=True, text=True, check=True,
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            return []

    @staticmethod
    def get_default_sink():
        """Return (id, node name, description) of the default output device, or None."""
        objects = AudioCapture._pw_dump()
        default_name = None
        for obj in objects:
            if obj.get("type") != "PipeWire:Interface:Metadata":
                continue
            for entry in obj.get("metadata", []):
                if entry.get("key") == "default.audio.sink":
                    default_name = (entry.get("value") or {}).get("name")
        for obj in objects:
            if obj.get("type") != "PipeWire:Interface:Node":
                continue
            props = obj.get("info", {}).get("props", {})
            if props.get("media.class") != "Audio/Sink":
                continue
            name = props.get("node.name", "")
            if default_name is None or name == default_name:
                return str(obj["id"]), name, props.get("node.description") or name
        return None

    @staticmethod
    def get_active_streams():
        """Return (id, node name, description) for sinks that are playing audio."""
        objects = AudioCapture._pw_dump()

        nodes = {}
        for obj in objects:
            if obj.get("type") != "PipeWire:Interface:Node":
                continue
            info = obj.get("info", {})
            props = info.get("props", {})
            nodes[obj["id"]] = {
                "name": props.get("node.name", ""),
                "description": props.get("node.description") or props.get("node.name", ""),
                "app": props.get("application.name") or props.get("node.name", ""),
                "media": props.get("media.class", ""),
                "state": info.get("state", ""),
            }

        apps_by_sink = {}
        for obj in objects:
            if obj.get("type") != "PipeWire:Interface:Link":
                continue
            info = obj.get("info", {})
            if info.get("state") != "active":
                continue
            source = nodes.get(info.get("output-node-id"))
            sink = nodes.get(info.get("input-node-id"))
            if not source or not sink:
                continue
            if source["media"] != "Stream/Output/Audio" or sink["media"] != "Audio/Sink":
                continue
            if source["state"] not in ("running", "idle"):
                continue
            apps_by_sink.setdefault(info["input-node-id"], set()).add(source["app"])

        streams = []
        for sink_id, apps in apps_by_sink.items():
            sink = nodes[sink_id]
            if not sink["name"] or sink["name"].startswith("input.loopback."):
                continue
            desc = f"{sink['description']} ({', '.join(sorted(apps))})"
            streams.append((str(sink_id), sink["name"], desc))
        return streams
    
    @staticmethod
    def capture_audio(source_name, segment_size):
        """Capture audio segment from a playing output device."""
        proc = subprocess.Popen(
            AudioCapture._record_command(source_name),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            audio_data = proc.stdout.read(segment_size)
        finally:
            proc.terminate()
            proc.wait()

        return audio_data


class ModelConfig:
    """Whisper model configurations optimized for live Chinese speech."""
    
    # vad_silence_ms    : silence duration (ms) that ends an utterance
    # vad_aggressiveness : webrtcvad aggressiveness 0 (permissive) → 3 (aggressive)
    # max_speech_ms      : force-flush after this duration even if no silence
    # beam               : Whisper beam search width (higher = better quality, slower)
    CONFIGS = {
        "tiny": {
            "fast":   {"beam": 1, "vad_silence_ms": 500,  "vad_aggressiveness": 3, "max_speech_ms": 2000, "precision": "60%"},
            "normal": {"beam": 1, "vad_silence_ms": 900,  "vad_aggressiveness": 2, "max_speech_ms": 4000, "precision": "60%"},
            "slow":   {"beam": 1, "vad_silence_ms": 1200, "vad_aggressiveness": 1, "max_speech_ms": 8000, "precision": "60%"},
        },
        "base": {
            "fast":   {"beam": 1, "vad_silence_ms": 500,  "vad_aggressiveness": 3, "max_speech_ms": 2000, "precision": "85%"},
            "normal": {"beam": 2, "vad_silence_ms": 900,  "vad_aggressiveness": 2, "max_speech_ms": 4000, "precision": "85%"},
            "slow":   {"beam": 3, "vad_silence_ms": 1200, "vad_aggressiveness": 1, "max_speech_ms": 8000, "precision": "85%"},
        },
        "small": {
            "fast":   {"beam": 1, "vad_silence_ms": 500,  "vad_aggressiveness": 3, "max_speech_ms": 2000, "precision": "90%"},
            "normal": {"beam": 2, "vad_silence_ms": 900,  "vad_aggressiveness": 2, "max_speech_ms": 4000, "precision": "90%"},
            "slow":   {"beam": 3, "vad_silence_ms": 1200, "vad_aggressiveness": 1, "max_speech_ms": 8000, "precision": "90%"},
        },
        "medium": {
            "fast":   {"beam": 2, "vad_silence_ms": 500,  "vad_aggressiveness": 3, "max_speech_ms": 2000, "precision": "95%"},
            "normal": {"beam": 3, "vad_silence_ms": 900,  "vad_aggressiveness": 2, "max_speech_ms": 4000, "precision": "95%"},
            "slow":   {"beam": 4, "vad_silence_ms": 1200, "vad_aggressiveness": 1, "max_speech_ms": 8000, "precision": "95%"},
        },
        "large-v3-turbo": {
            "fast":   {"beam": 1, "vad_silence_ms": 500,  "vad_aggressiveness": 3, "max_speech_ms": 3000, "precision": "high"},
            "normal": {"beam": 2, "vad_silence_ms": 900,  "vad_aggressiveness": 2, "max_speech_ms": 6000, "precision": "high"},
            "slow":   {"beam": 3, "vad_silence_ms": 1200, "vad_aggressiveness": 1, "max_speech_ms": 10000, "precision": "high"},
        },
        "large-v3": {
            "fast":   {"beam": 2, "vad_silence_ms": 500,  "vad_aggressiveness": 3, "max_speech_ms": 2000, "precision": "98%"},
            "normal": {"beam": 3, "vad_silence_ms": 900,  "vad_aggressiveness": 2, "max_speech_ms": 4000, "precision": "98%"},
            "slow":   {"beam": 4, "vad_silence_ms": 1200, "vad_aggressiveness": 1, "max_speech_ms": 8000, "precision": "98%"},
        },
    }

    MODE_DESC = {
        "fast":   "500ms pause — reactive, may cut mid-sentence",
        "normal": "800ms pause — balanced",
        "slow":   "1200ms pause — waits for complete sentences",
    }
    
    @classmethod
    def get_config(cls, model, mode="normal"):
        """Get configuration for model and mode"""
        return cls.CONFIGS.get(model, cls.CONFIGS["medium"])[mode]


class TranscriptWriter:
    """Handle saving transcripts to Markdown files"""

    def __init__(self, filepath, model, mode):
        self.filepath = filepath
        self.file_handle = None
        self.target_flag = TARGET_FLAG

        if filepath:
            try:
                self.file_handle = open(filepath, 'w', encoding='utf-8')
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.file_handle.write(f"# Live Voice Translation\n\n")
                self.file_handle.write(f"**Date:** {now}  \n")
                self.file_handle.write(f"**Model:** {model}  \n")
                self.file_handle.write(f"**Mode:** {mode}  \n")
                self.file_handle.write(f"**Language:** {SOURCE_LABEL} → {TARGET_LABEL}  \n\n")
                self.file_handle.write("---\n\n")
                self.file_handle.flush()
                print(f"💾 Saving to: {filepath}\n", flush=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not open save file: {e}\n", flush=True)
                self.file_handle = None

    def write(self, timestamp, text_it, text_target):
        """Write translation to file"""
        if self.file_handle:
            self.file_handle.write(f"**[{timestamp}]**\n\n")
            self.file_handle.write(f"{SOURCE_FLAG} *{text_it}*\n\n")
            self.file_handle.write(f"{self.target_flag} {text_target}\n\n")
            self.file_handle.write("---\n\n")
            self.file_handle.flush()
    
    def close(self, duration_str=None, phrase_count=None, word_count=None, dropped_count=None):
        """Close file with end timestamp and optional session statistics"""
        if self.file_handle:
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.file_handle.write(f"\n---\n\n")
            self.file_handle.write(f"**End of session:** {end_time}  \n")
            if duration_str is not None:
                self.file_handle.write(f"**Duration:** {duration_str}  \n")
                self.file_handle.write(f"**Phrases:** {phrase_count}  \n")
                self.file_handle.write(f"**Words:** {word_count}  \n")
                if dropped_count:
                    self.file_handle.write(f"**Dropped:** {dropped_count} ⚠️ CPU too slow for this model  \n")
            self.file_handle.close()
            print(f"\n💾 Transcript saved to: {self.filepath}")


class Overlay:
    """Always-on-top window showing the last sentence: source on top, translation below.

    The panel background is half transparent and the text stays opaque. Drag
    anywhere except the close mark; the window manager performs the move.
    """

    WIDTH = 920
    HEIGHT = 240
    # (path, face index) candidates; the font must cover both CJK and Latin glyphs.
    _FONTS = (
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),  # Noto Sans CJK SC
        ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 2),
        ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc", 2),
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
    )
    _MAX_FONT_SIZE = 30
    _MIN_FONT_SIZE = 14
    _PADDING = 20

    def __init__(self):
        self._source = ""
        self._target = ""
        self._font_file = next(
            ((path, index) for path, index in self._FONTS if Path(path).is_file()),
            None,
        )
        self._closed = False
        self._glfw = None
        self._gl = None
        self._window = None
        self._texture = None
        self._texture_key = None
        # Set by the translator: on_key(char) handles a shortcut pressed while
        # the overlay has focus; status() returns (mode, paused) for the footer.
        self.on_key = None
        self.status = None

    def open(self):
        # GLFW can only move and composite a transparent window through X11.
        os.environ.pop("WAYLAND_DISPLAY", None)
        import glfw
        from OpenGL import GL

        self._glfw = glfw
        self._gl = GL
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
        if not glfw.init():
            raise RuntimeError("GLFW failed to initialize")

        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
        self._window = glfw.create_window(
            self.WIDTH, self.HEIGHT, "live-voice-translate", None, None,
        )
        if self._window is None:
            glfw.terminate()
            raise RuntimeError("Could not create the overlay window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)
        glfw.set_mouse_button_callback(self._window, self._on_mouse)
        glfw.set_key_callback(self._window, self._on_key)
        video = glfw.get_video_mode(glfw.get_primary_monitor())
        screen_h = video.size[1] if video is not None else 1080
        glfw.set_window_pos(self._window, 40, max(40, screen_h - self.HEIGHT - 80))
        self.pump()

    def push(self, source, target):
        source = " ".join(source.split())
        target = " ".join(target.split())
        if source or target:
            self._source, self._target = source, target

    def pump(self):
        """Draw one frame. Return False when the window has been closed."""
        glfw = self._glfw
        if glfw is None or self._window is None or self._closed:
            self._closed = True
            return False
        glfw.poll_events()
        if glfw.window_should_close(self._window):
            self._closed = True
            return False
        self._draw()
        glfw.swap_buffers(self._window)
        return True

    def close(self):
        glfw = self._glfw
        self._closed = True
        if glfw is not None and self._window is not None:
            glfw.destroy_window(self._window)
            glfw.terminate()
        self._window = None
        self._glfw = None

    def _shown_text(self):
        if not (self._source or self._target):
            return "Waiting for speech…", ""
        return self._source, self._target

    def _font(self, size):
        from PIL import ImageFont

        if self._font_file is None:
            return ImageFont.load_default(size)
        path, index = self._font_file
        return ImageFont.truetype(path, size, index=index)

    def _fit(self, draw, text, max_width, max_height):
        """Largest font whose wrapped text fits the box, with its lines and line height."""
        for size in range(self._MAX_FONT_SIZE, self._MIN_FONT_SIZE - 1, -2):
            font = self._font(size)
            lines = _wrap_text(draw, text, font, max_width)
            line_height = int(size * 1.35)
            if len(lines) * line_height <= max_height:
                return font, lines, line_height
        max_lines = max(1, max_height // line_height)
        return font, lines[-max_lines:], line_height

    def _on_key(self, window, key, scancode, action, mods):
        """Forward letter shortcuts typed into the overlay to the translator."""
        glfw = self._glfw
        if action != glfw.PRESS or self.on_key is None:
            return
        name = glfw.get_key_name(key, scancode)
        if name and len(name) == 1 and name.isalpha():
            self.on_key(name.lower())

    def _status_line(self):
        if self.status is None:
            return ""
        mode, paused = self.status()
        state = "PAUSED" if paused else f"mode: {mode}"
        return f"{state}    M mode   P pause   Q quit"

    def _on_mouse(self, window, button, action, mods):
        glfw = self._glfw
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        width, _height = glfw.get_window_size(window)
        x_pos, y_pos = glfw.get_cursor_pos(window)
        if x_pos >= width - 42 and y_pos <= 42:
            glfw.set_window_should_close(window, True)
            return
        self._begin_move(x_pos, y_pos)

    def _begin_move(self, x_pos, y_pos):
        """Ask the window manager to drag the window from this click."""
        glfw = self._glfw
        xid = glfw.get_x11_window(self._window)
        if not xid:
            return
        origin_x, origin_y = glfw.get_window_pos(self._window)
        from Xlib import X, display
        from Xlib.protocol.event import ClientMessage

        dpy = display.Display()
        window = dpy.create_resource_object("window", xid)
        atom = dpy.intern_atom("_NET_WM_MOVERESIZE")
        message = ClientMessage(
            window=window,
            client_type=atom,
            data=(32, [int(origin_x + x_pos), int(origin_y + y_pos), 8, 1, 1]),
        )
        dpy.screen().root.send_event(
            message,
            event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask,
        )
        dpy.flush()

    def _draw(self):
        GL = self._gl
        glfw = self._glfw
        width, height = glfw.get_framebuffer_size(self._window)
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.0, 0.0, 0.0, 0.5)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self._draw_text(width, height)

    def _draw_text(self, width, height):
        GL = self._gl
        key = (self._shown_text(), self._status_line(), width, height)
        if key != self._texture_key:
            self._upload_text(width, height)
            self._texture_key = key
        if self._texture is None:
            return

        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glColor4f(1.0, 1.0, 1.0, 1.0)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, width, height, 0, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(0, 0)
        GL.glTexCoord2f(1, 0); GL.glVertex2f(width, 0)
        GL.glTexCoord2f(1, 1); GL.glVertex2f(width, height)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(0, height)
        GL.glEnd()

    def _upload_text(self, width, height):
        image = self._render(width, height)
        GL = self._gl
        if self._texture is None:
            self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, image.tobytes(),
        )

    def _render(self, width, height):
        """Draw the text panel into a transparent RGBA image."""
        from PIL import Image, ImageDraw

        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        pad = self._PADDING
        status = self._status_line()
        footer_h = 30 if status else 0
        half = (height - footer_h) // 2
        text_width = width - 2 * pad - 30  # leave room for the close mark
        source, target = self._shown_text()

        halves = (
            (source, 0, (150, 230, 160, 255)),
            (target, half, (255, 255, 255, 255)),
        )
        for text, top, color in halves:
            if not text:
                continue
            font, lines, line_height = self._fit(draw, text, text_width, half - 2 * pad)
            y_pos = top + (half - len(lines) * line_height) // 2
            for line in lines:
                draw.text((pad, y_pos), line, font=font, fill=color)
                y_pos += line_height

        draw.line((pad, half, width - pad, half), fill=(255, 255, 255, 70), width=1)
        draw.text((width - 32, 6), "×", font=self._font(24), fill=(255, 255, 255, 210))
        if status:
            paused = status.startswith("PAUSED")
            color = (255, 200, 90, 255) if paused else (255, 255, 255, 150)
            draw.text((pad, height - footer_h + 4), status, font=self._font(16), fill=color)
        return image


_WRAP_TOKEN_RE = re.compile(r'[\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]|[^\s\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]+|\s+')


def _wrap_text(draw, text, font, max_width):
    """Wrap at spaces for Latin text and between characters for Chinese."""
    lines, current = [], ""
    for token in _WRAP_TOKEN_RE.findall(text):
        trial = current + token
        if not current.strip() or draw.textlength(trial.rstrip(), font=font) <= max_width:
            current = trial
        else:
            lines.append(current.rstrip())
            current = token.lstrip()
    if current.strip():
        lines.append(current.rstrip())
    return lines


class LiveTranslator:
    """Main translation engine with keyboard control"""
    
    def __init__(self, model_name, mode, save_file=None, enable_keyboard=True, show_chinese=False, vad_filter=True, use_gpu=False, mixed=False, overlay=False):
        self.model_name = model_name
        self.mode = mode
        self.mixed = mixed
        self.overlay = Overlay() if overlay else None
        self.config = ModelConfig.get_config(model_name, mode)
        self.writer = TranscriptWriter(save_file, model_name, mode)
        self.model = None
        self.vad_filter = vad_filter
        self.use_gpu = use_gpu

        # Keyboard control
        self.enable_keyboard = enable_keyboard
        self.keyboard_controller = None
        self.paused = False
        self.should_quit = False

        # Display options
        self.show_chinese = show_chinese
        
        # Audio threading
        self.audio_queue = queue.Queue(maxsize=2)
        self.capture_thread = None

        # Rolling context passed to Whisper as initial_prompt (last ~200 Chinese characters)
        self._whisper_context = ""

        # Sentence accumulation buffer: holds fragments until a complete sentence is ready
        self._sentence_buffer = ""
        self._sentence_buffer_time = None  # time when buffer was last updated

        # Recent translated phrases for dedup (last 5 outputs)
        self._recent_phrases: collections.deque = collections.deque(maxlen=5)

        # Recent Chinese segments for dedup (last 5 inputs)
        self._recent_chinese: collections.deque = collections.deque(maxlen=5)

        # Session statistics
        self.session_start = datetime.now()
        self.phrase_count = 0
        self.word_count = 0
        self.dropped_count = 0
        
    def setup(self):
        """Initialize Whisper model, translation, and keyboard"""
        # Ensure required translation models are installed
        self._install_translation_model()
        
        # Load Whisper model
        print(f"⏳ Loading model {self.model_name}...", flush=True)
        if self.use_gpu:
            print(f"⏳ Loading on GPU ({cuda_device_name()})...", flush=True)
            try:
                self.model = WhisperModel(self.model_name, device="cuda", compute_type="float16")
                print("✅ GPU loaded\n", flush=True)
            except Exception as e:
                print(f"⚠️  GPU failed ({e}), falling back to CPU", flush=True)
                self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                self.use_gpu = False
        else:
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        print("✅ Ready\n", flush=True)
        
        # Start keyboard controller
        if self.enable_keyboard:
            self.keyboard_controller = KeyboardController(self)
            if not self.keyboard_controller.setup():
                self.keyboard_controller = None
    
    @staticmethod
    def _install_translation_model():
        """Ensure the direct Chinese→English Argos model is installed."""
        try:
            installed_packages = argostranslate.package.get_installed_packages()
            if any(
                package.from_code == SOURCE_LANG
                and package.to_code == TARGET_LANG
                for package in installed_packages
            ):
                return

            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                (
                    package
                    for package in available_packages
                    if package.from_code == SOURCE_LANG
                    and package.to_code == TARGET_LANG
                ),
                None,
            )
            if package_to_install is None:
                raise RuntimeError(
                    f"No Argos model found for {SOURCE_LANG}→{TARGET_LANG}"
                )

            print(
                f"⏳ Installing translation model {SOURCE_LANG}→{TARGET_LANG}...",
                flush=True,
            )
            argostranslate.package.install_from_path(package_to_install.download())
            print(f"✅ {SOURCE_LANG}→{TARGET_LANG} installed", flush=True)
        except Exception as e:
            raise RuntimeError(
                f"Could not install the Chinese→English translation model: {e}"
            ) from e
    
    def _audio_capture_loop(self, source_name):
        """Background thread: continuous audio capture with VAD-based segmentation"""
        SAMPLE_RATE = 16000
        FRAME_MS = 30                                     # 30 ms per VAD frame
        FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 480 samples
        FRAME_BYTES = FRAME_SAMPLES * 2                   # 960 bytes (16-bit mono)

        PRE_PADDING_FRAMES = 300 // FRAME_MS  # ~10 frames kept before speech starts
        MIN_SPEECH_FRAMES = 500 // FRAME_MS   # ~17 frames = 500 ms minimum utterance

        current_aggressiveness = self.config["vad_aggressiveness"]
        vad = webrtcvad.Vad(current_aggressiveness)

        proc = None
        ring_buffer = collections.deque(maxlen=PRE_PADDING_FRAMES)
        voiced_frames = []
        in_speech = False
        silence_count = 0

        def start_recorder():
            return subprocess.Popen(
                AudioCapture._record_command(source_name),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

        try:
            proc = start_recorder()
            while not self.should_quit:
                frame = proc.stdout.read(FRAME_BYTES)
                if not frame or len(frame) < FRAME_BYTES:
                    # pw-record exited (device removed, PipeWire restarted...).
                    # Keep listening: restart it instead of ending the session.
                    proc.terminate()
                    proc.wait()
                    if self.should_quit:
                        break
                    print("\n⚠️  Audio capture stopped, reconnecting...", flush=True)
                    time.sleep(1.0)
                    in_speech = False
                    voiced_frames = []
                    ring_buffer.clear()
                    silence_count = 0
                    proc = start_recorder()
                    continue

                if self.paused:
                    in_speech = False
                    voiced_frames = []
                    ring_buffer.clear()
                    silence_count = 0
                    continue

                # Reload VAD params from config (updated when mode changes)
                new_aggressiveness = self.config["vad_aggressiveness"]
                if new_aggressiveness != current_aggressiveness:
                    current_aggressiveness = new_aggressiveness
                    vad = webrtcvad.Vad(current_aggressiveness)
                silence_frames = self.config["vad_silence_ms"] // FRAME_MS

                try:
                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    is_speech = False

                if not in_speech:
                    ring_buffer.append(frame)
                    if is_speech:
                        in_speech = True
                        voiced_frames = list(ring_buffer)
                        silence_count = 0
                else:
                    voiced_frames.append(frame)
                    if not is_speech:
                        silence_count += 1
                        if silence_count >= silence_frames:
                            # End of utterance — flush to queue
                            if len(voiced_frames) >= MIN_SPEECH_FRAMES:
                                audio_data = b"".join(voiced_frames)
                                try:
                                    self.audio_queue.put(audio_data, timeout=0.5)
                                except queue.Full:
                                    self.dropped_count += 1
                            in_speech = False
                            voiced_frames = []
                            ring_buffer.clear()
                            silence_count = 0
                    else:
                        silence_count = 0
                        # Force-flush if utterance exceeds the mode's max duration
                        max_speech_frames = self.config["max_speech_ms"] // FRAME_MS
                        if len(voiced_frames) >= max_speech_frames:
                            audio_data = b"".join(voiced_frames)
                            try:
                                self.audio_queue.put(audio_data, timeout=0.5)
                            except queue.Full:
                                self.dropped_count += 1
                            # Keep last 500ms as overlap to avoid cutting words at chunk boundary
                            overlap_frames = 500 // FRAME_MS
                            voiced_frames = list(voiced_frames[-overlap_frames:])
        except Exception as e:
            if not self.should_quit:
                print(f"\n⚠️  Audio capture error: {e}", flush=True)
        finally:
            if proc is not None:
                proc.terminate()
                proc.wait()
    
    def process_audio(self, audio_data):
        """Transcribe and translate audio segment (with pause support)"""
        # Check if paused
        if self.paused:
            return

        # Flush sentence buffer if it has been waiting too long (1.5s timeout)
        if self._sentence_buffer and self._sentence_buffer_time is not None:
            if time.time() - self._sentence_buffer_time > 1.5:
                clean = re.sub(r'\s*\.\.\.\s*', ' ', self._sentence_buffer).strip()
                self._translate_and_display(clean, skip_source=True)
                self._sentence_buffer = ""
                self._sentence_buffer_time = None
        
        # Skip very short audio (< 300 ms)
        if len(audio_data) < 9600:
            return
        
        # faster-whisper expects 16 kHz mono float32 samples in [-1, 1]
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe with Whisper
        # initial_prompt feeds the last ~200 Chinese characters to Whisper
        # context across VAD chunks: better capitalisation, punctuation, continuity
        # word_timestamps=True lets us stream Chinese tokens as they are decoded
        # In mixed mode Whisper detects the language of each segment instead of
        # forcing Chinese, so English speech is transcribed as English.
        segments, _ = self.model.transcribe(
            audio,
            language=None if self.mixed else SOURCE_LANG,
            multilingual=self.mixed,
            vad_filter=self.vad_filter,
            beam_size=self.config["beam"],
            condition_on_previous_text=False,
            initial_prompt=self._whisper_context or None,
            word_timestamps=True,
        )

        # Translate and display
        separator_printed = False
        chunk_chinese = []
        for segment in segments:
            # Drop hallucinated segments: Whisper loops on silence/noise
            # no_speech_prob > 0.6 means Whisper is not confident speech was heard
            # compression_ratio > 2.4 means the output is suspiciously repetitive
            if getattr(segment, 'no_speech_prob', 0.0) > 0.6:
                continue
            if getattr(segment, 'compression_ratio', 1.0) > 2.4:
                continue
            text_it = segment.text.strip()
            if text_it and len(text_it) >= 2:
                # Skip entire segment if identical to a recently seen one
                text_it_norm = text_it.lower()
                if text_it_norm in self._recent_chinese:
                    continue
                self._recent_chinese.append(text_it_norm)

                # Remove Whisper loop artefacts within the segment text
                text_it = self._clean_hallucination_loops(text_it)

                chunk_chinese.append(text_it)
                if not separator_printed:
                    print("─" * 50, flush=True)
                    separator_printed = True

                # Stream Chinese tokens as they are decoded.
                if self.show_chinese and segment.words:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"\033[92m[{timestamp}]", end=" ", flush=True)
                    for word_info in segment.words:
                        print(word_info.word, end="", flush=True)
                        time.sleep(0.02)
                    print("\033[0m", flush=True)

                # Accumulate fragments and only translate complete sentences
                # Chinese normally has no whitespace after 。！？, unlike Latin text.
                for part in re.split(r'(?<=[.!?。！？])\s*', text_it):
                    part = part.strip()
                    if not part or len(part) < 2:
                        continue
                    # Don't merge an English sentence into a Chinese one (or vice versa):
                    # the translate/pass-through decision is made per flushed buffer.
                    if (self.mixed and self._sentence_buffer
                            and is_chinese(part) != is_chinese(self._sentence_buffer)):
                        self._translate_and_display(self._sentence_buffer, skip_source=True)
                        self._sentence_buffer = ""
                        self._sentence_buffer_time = None
                    self._sentence_buffer = (self._sentence_buffer + " " + part).strip() if self._sentence_buffer else part
                    self._sentence_buffer_time = time.time()
                    # Flush when the accumulated buffer ends with terminal punctuation (not ellipsis)
                    ends_sentence = bool(re.search(r'[.!?。！？]\s*$', self._sentence_buffer)) and not self._sentence_buffer.rstrip().endswith('...')
                    too_long = len(self._sentence_buffer) > 200
                    if ends_sentence or too_long:
                        clean = re.sub(r'\s*\.\.\.\s*', ' ', self._sentence_buffer).strip()
                        self._translate_and_display(clean, skip_source=True)
                        self._sentence_buffer = ""
                        self._sentence_buffer_time = None

        # Update rolling context for next chunk (keep last ~200 chars)
        if chunk_chinese:
            combined = " ".join(chunk_chinese)
            self._whisper_context = (self._whisper_context + " " + combined)[-200:].lstrip()
    
    @staticmethod
    def _clean_hallucination_loops(text: str) -> str:
        """Remove Whisper loop artefacts: phrases repeated 3+ times in a row.

        Whisper sometimes generates 'X, X, X, X...' when it loses track of
        the audio. Three or more consecutive repetitions of the same clause
        (separated by commas) are collapsed to a single occurrence.
        Two repetitions are kept — they may be intentional emphasis.
        """
        return re.sub(
            r'(\b\w[\w\'\s]{2,40}?)(?:,\s*\1){2,}',
            r'\1',
            text,
            flags=re.IGNORECASE,
        ).strip()

    def _translate_and_display(self, text_it, skip_source=False):
        """Translate Chinese text to English and display/save."""
        if self.mixed and not is_chinese(text_it):
            text_target = text_it
        else:
            text_target = argostranslate.translate.translate(
                text_it, SOURCE_LANG, TARGET_LANG
            )

        # Skip exact duplicate of any recently displayed phrase (Whisper loop guard)
        text_target_norm = text_target.strip().lower()
        if text_target_norm in self._recent_phrases:
            return
        self._recent_phrases.append(text_target_norm)
        if self.overlay:
            self.overlay.push(text_it, text_target)

        # Get timestamp and flag
        timestamp = datetime.now().strftime('%H:%M:%S')
        # Display Chinese if enabled and not already streamed token by token.
        if self.show_chinese and not skip_source:
            if len(text_it) > 70:
                output_it = textwrap.fill(
                    text_it, width=70,
                    initial_indent=f"[{timestamp}] ",
                    subsequent_indent=" " * 13
                )
            else:
                output_it = f"[{timestamp}] {text_it}"
            print(f"\033[92m{output_it}\033[0m", flush=True)

        # Display translated text word by word with 20ms micro-delay (typewriter effect)
        words = text_target.split()
        print(f"[{timestamp}] ▶ ", end="", flush=True)
        for word in words:
            print(word + " ", end="", flush=True)
            time.sleep(0.02)
        print(flush=True)

        # Update session stats
        self.phrase_count += 1
        self.word_count += len(text_target.split())

        # Save to file if enabled
        self.writer.write(timestamp, text_it, text_target)
    
    def run(self, stream_id):
        """Main translation loop with keyboard control"""
        # Start audio capture in background thread
        self.capture_thread = Thread(
            target=self._audio_capture_loop, 
            args=(stream_id,), 
            daemon=True
        )
        self.capture_thread.start()

        if self.overlay:
            try:
                # Shortcuts also work when the overlay window has focus.
                self.overlay.on_key = (self.keyboard_controller or KeyboardController(self)).handle_key
                self.overlay.status = lambda: (self.mode, self.paused)
                self.overlay.open()
            except Exception as exc:
                print(f"⚠️  Overlay failed to open: {exc}", flush=True)
                self.overlay.close()
                self.overlay = None
        
        try:
            while not self.should_quit:
                if self.overlay and not self.overlay.pump():
                    self.should_quit = True
                    break

                # Check keyboard input (fast, non-blocking)
                if self.keyboard_controller:
                    key = self.keyboard_controller.get_key(timeout=0.01)
                    if key:
                        self.keyboard_controller.handle_key(key)
                
                # Get audio from queue (non-blocking with timeout)
                try:
                    audio_data = self.audio_queue.get(timeout=0.02 if self.overlay else 0.1)
                    self.process_audio(audio_data)
                except queue.Empty:
                    # No audio ready yet, continue checking keyboard
                    pass
                    
        except KeyboardInterrupt:
            print("\n\nStopping translation.")
        finally:
            # Signal threads to stop
            self.should_quit = True
            if self.overlay:
                self.overlay.close()

            # Cleanup keyboard
            if self.keyboard_controller:
                self.keyboard_controller.cleanup()

            # Wait for capture thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)

            # Print session statistics
            duration = datetime.now() - self.session_start
            total_seconds = int(duration.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            print()
            print("═══════════════════════════════════════════")
            print("   Session Statistics")
            print("═══════════════════════════════════════════")
            print(f"  Duration   : {duration_str}")
            print(f"  Phrases    : {self.phrase_count}")
            print(f"  Words      : {self.word_count}")
            if self.dropped_count > 0:
                print(f"  Dropped    : {self.dropped_count}  ⚠️  Consider --fast or a smaller model")
            else:
                print(f"  Dropped    : 0")
            print("═══════════════════════════════════════════")
            print()

            # Close file (with session stats if saving)
            self.writer.close(
                duration_str=duration_str,
                phrase_count=self.phrase_count,
                word_count=self.word_count,
                dropped_count=self.dropped_count,
            )


_CJK_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
_LATIN_WORD_RE = re.compile(r'[A-Za-z]+')


def is_chinese(text):
    """Whether text is mainly Chinese.

    One Chinese character carries roughly one word, so characters are compared
    against Latin words: "这个 feature 我们下周 release" counts as Chinese.
    """
    return len(_CJK_RE.findall(text)) >= max(1, len(_LATIN_WORD_RE.findall(text)))


def cuda_available():
    """Return whether CTranslate2 can see an NVIDIA GPU."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def cuda_device_name():
    """Return the first NVIDIA device name, or a generic CUDA label."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=3,
        )
        name = result.stdout.strip().splitlines()[0].strip()
        return name or "CUDA"
    except Exception:
        return "CUDA"


def show_menu():
    """Interactive model selection menu"""
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
        "": "large-v3-turbo"
    }
    
    return models.get(choice, "large-v3-turbo")


def main():
    """Main entry point"""
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
        """
    )
    
    parser.add_argument(
        "model",
        nargs="?",
        choices=["tiny", "base", "small", "medium", "turbo", "large"],
        help="Whisper model to use"
    )
    
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="500ms silence threshold, reduced beam — reactive, may cut mid-sentence"
    )

    parser.add_argument(
        "--slow", "-s",
        action="store_true",
        help="1200ms silence threshold, max beam — waits for complete sentences"
    )
    
    parser.add_argument(
        "--save",
        nargs="?",
        const=f"live-translate-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md",
        metavar="FILE",
        help="Save transcript to file (default: auto-generated)"
    )
    
    parser.add_argument(
        "--show-chinese", "--show-italian", "-i",
        dest="show_chinese",
        action="store_true",
        help="Display Chinese source text (toggle with 'i' key)"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Transcribe on the NVIDIA GPU (this is the default when CUDA is available)"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU transcription"
    )

    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Audio mixes Chinese and English: translate Chinese parts, show English parts as is"
    )

    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Floating half-transparent window with the last sentence and its translation"
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Whisper's internal VAD filter (webrtcvad chunking is always active)"
    )

    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable keyboard shortcuts"
    )
    
    args = parser.parse_args()

    if args.cpu and args.gpu:
        print("Choose either --gpu or --cpu.", file=sys.stderr)
        sys.exit(2)
    use_gpu = False if args.cpu else args.gpu or cuda_available()
    
    # Determine model
    if args.model:
        model = {
            "large": "large-v3",
            "turbo": "large-v3-turbo",
        }.get(args.model, args.model)
    else:
        model = show_menu()
    
    # Determine mode
    if args.fast:
        mode = "fast"
    elif args.slow:
        mode = "slow"
    else:
        mode = "normal"
    
    # Get model config for display
    config = ModelConfig.get_config(model, mode)
    
    # Display configuration
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
        print(f"  Chinese    : Displayed")
    source = f"{SOURCE_LABEL} + {TARGET_LABEL}" if args.mixed else SOURCE_LABEL
    print(f"  Language   : {source} → {TARGET_LABEL}")
    print(f"  Device     : {cuda_device_name() if use_gpu else 'CPU'}")
    print(f"  VAD        : {'Disabled' if args.no_vad else 'Enabled (skips silence)'}")
    if args.overlay:
        print("  Overlay    : Last sentence (source + translation)")
    print()
    
    # Detect audio streams
    streams = AudioCapture.get_active_streams()
    if not streams:
        # Nothing is playing yet: listen to the default output device and let
        # the VAD wait for speech.
        default_sink = AudioCapture.get_default_sink()
        if default_sink is None:
            print("Error: No PipeWire audio output device found", file=sys.stderr)
            sys.exit(1)
        _, stream_id, stream_desc = default_sink
        stream_desc += " (no audio playing yet, waiting)"
    elif len(streams) == 1:
        _, stream_id, stream_desc = streams[0]
    else:
        print("  Multiple audio streams detected:")
        print()
        for i, (sid, sname, desc) in enumerate(streams, 1):
            print(f"    {i}) {desc}")
        print()
        choice = input(f"  Select stream (1-{len(streams)}): ").strip()
        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(streams)):
                raise ValueError
        except ValueError:
            print("Invalid choice, using first stream.")
            idx = 0
        _, stream_id, stream_desc = streams[idx]

    print(f"  Stream     : {stream_desc}")
    print()
    print("═══════════════════════════════════════════")
    print()
    print("Translation in progress... (Ctrl+C to stop)")
    print()
    
    # Run translator
    translator = LiveTranslator(
        model, mode, args.save,
        enable_keyboard=not args.no_keyboard,
        show_chinese=args.show_chinese,
        mixed=args.mixed,
        vad_filter=not args.no_vad,
        use_gpu=use_gpu,
        overlay=args.overlay,
    )
    translator.setup()
    translator.run(stream_id)


if __name__ == "__main__":
    main()
