#!/usr/bin/env python3
"""
live-voice-translate - Real-time Italian→English audio translation
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

import os
import sys
import subprocess
from pathlib import Path

# ============================================================================
# VIRTUALENV SETUP - Runs first, before any other imports
# ============================================================================

VENV_DIR = Path.home() / ".local/share/live-voice-translate/venv"
REQUIRED_PACKAGES = ["faster-whisper", "argostranslate", "webrtcvad"]
MIN_PYTHON = (3, 9)


def detect_distro():
    """Return a distro family string based on /etc/os-release."""
    try:
        info = {}
        with open("/etc/os-release") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k] = v.strip('"').lower()
        id_ = info.get("ID", "")
        like = info.get("ID_LIKE", "")
        if id_ == "fedora" or "rhel" in like or "fedora" in like or "centos" in like:
            return "fedora"
        if "suse" in id_ or "suse" in like:
            return "opensuse"
        if id_ in ("ubuntu", "debian", "linuxmint", "pop") or "debian" in like or "ubuntu" in like:
            return "debian"
    except Exception:
        pass
    return "unknown"


def find_compatible_python():
    """Return path to a Python >= MIN_PYTHON executable, or None."""
    import shutil
    names = [f"python3.{minor}" for minor in range(13, 8, -1)]
    # shutil.which can silently fail on Python 3.6 — also probe common prefixes directly
    search_prefixes = ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]
    candidates = []
    for name in names:
        path = shutil.which(name)
        if path and path not in candidates:
            candidates.append(path)
        for prefix in search_prefixes:
            full = f"{prefix}/{name}"
            if full not in candidates and os.path.isfile(full) and os.access(full, os.X_OK):
                candidates.append(full)
    for path in candidates:
        try:
            result = subprocess.run(
                [path, "-c",
                 f"import sys; sys.exit(0 if sys.version_info >= {MIN_PYTHON} else 1)"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return path
        except Exception:
            continue
    return None


def check_python_version():
    """Exit with a helpful message if Python is too old."""
    if sys.version_info >= MIN_PYTHON:
        return

    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    req = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    print(f"⚠️  Python {ver} détecté — Python {req}+ requis.")
    print(f"🔍 Recherche d'une version compatible...", flush=True)

    found = find_compatible_python()
    if found:
        print(f"✅ {found} trouvé — relancement...\n", flush=True)
        os.execv(found, [found] + sys.argv)

    # No compatible Python found — print distro-specific instructions
    distro = detect_distro()
    print(f"❌ Aucun Python {req}+ trouvé sur ce système.\n")
    print("   Installez une version compatible :")
    print()
    if distro == "fedora":
        print("     sudo dnf install python3.11")
        print("     python3.11 ./lvt.py")
    elif distro == "opensuse":
        print("     sudo zypper install python311")
        print("     python3.11 ./lvt.py")
    elif distro == "debian":
        print("     sudo apt install python3.11")
        print("     python3.11 ./lvt.py")
    else:
        print("     Fedora/RHEL:   sudo dnf install python3.11")
        print("     openSUSE:      sudo zypper install python311")
        print("     Ubuntu/Debian: sudo apt install python3.11")
        print("     pyenv:         pyenv install 3.11.9 && pyenv local 3.11.9")
        print()
        print("     Puis relancez : python3.11 ./lvt.py")
    print()
    sys.exit(1)


def webrtcvad_devel_hint():
    """Print distro-specific install hint for Python development headers."""
    distro = detect_distro()
    ver = f"{sys.version_info.major}{sys.version_info.minor}"  # e.g. "311"
    print("   webrtcvad requires Python development headers.")
    print("   Install them for your distro, then delete the venv and re-run:")
    print(f"     rm -rf {VENV_DIR}")
    if distro == "fedora":
        print(f"     sudo dnf install python3-devel")
    elif distro == "opensuse":
        print(f"     sudo zypper install python{ver}-devel")
    elif distro == "debian":
        print(f"     sudo apt install python3-dev")
    else:
        print(f"     Fedora/RHEL:   sudo dnf install python3-devel")
        print(f"     openSUSE:      sudo zypper install python{ver}-devel")
        print(f"     Ubuntu/Debian: sudo apt install python3-dev")
    print()


def setup_virtualenv():
    """Create virtualenv and install dependencies if needed"""
    venv_python = VENV_DIR / "bin" / "python"
    pip = VENV_DIR / "bin" / "pip"

    # Reliable venv detection: sys.prefix points to the venv root when activated.
    # Comparing executable paths is wrong on systems where the venv Python is a
    # symlink to the system Python (e.g. Fedora), which makes both paths identical
    # even though site-packages are different.
    in_venv = Path(sys.prefix).resolve() == VENV_DIR.resolve()

    if VENV_DIR.exists():
        if in_venv:
            # Already running inside the venv — verify all packages are present
            missing = []
            for pkg in REQUIRED_PACKAGES:
                try:
                    result = subprocess.run(
                        [str(pip), "show", pkg],
                        capture_output=True,
                        timeout=10
                    )
                    if result.returncode != 0:
                        missing.append(pkg)
                except Exception:
                    missing.append(pkg)

            # Install missing packages
            if missing:
                print(f"\n⚠️  Installing missing packages: {', '.join(missing)}\n")
                for pkg in missing:
                    print(f"  Installing {pkg}...", flush=True)
                    try:
                        subprocess.run([str(pip), "install", pkg], check=True)
                        print(f"  ✅ {pkg} installed\n", flush=True)
                    except subprocess.CalledProcessError as e:
                        print(f"  ❌ Failed to install {pkg}: {e}\n", flush=True)
                        if pkg == "webrtcvad":
                            webrtcvad_devel_hint()
                        sys.exit(1)

            # All packages present, continue normally
            return

        # Venv exists but we're not inside it — re-execute with venv Python
        print(f"🔄 Switching to virtualenv Python...\n", flush=True)
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    
    # Venv doesn't exist, create it
    print("═══════════════════════════════════════════")
    print("   First Run Setup")
    print("═══════════════════════════════════════════")
    print(f"\nCreating isolated environment...")
    print(f"Location: {VENV_DIR}\n")
    
    # Create virtualenv
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtualenv: {e}")
        print("   Make sure python3-venv is installed:")
        print("   sudo dnf install python3-venv")
        sys.exit(1)
    
    # Install dependencies
    pip = VENV_DIR / "bin" / "pip"
    print(f"Installing dependencies: {', '.join(REQUIRED_PACKAGES)}")
    print("This may take 2-3 minutes...\n")
    
    # Upgrade pip and install wheel first (needed to build webrtcvad C extension)
    subprocess.run(
        [str(pip), "install", "--upgrade", "pip", "wheel"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    
    # Install packages one by one with progress
    for pkg in REQUIRED_PACKAGES:
        print(f"  Installing {pkg}...", flush=True)
        try:
            subprocess.run([str(pip), "install", pkg], check=True)
            print(f"  ✅ {pkg} installed\n", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {pkg}: {e}\n", flush=True)
            if pkg == "webrtcvad":
                webrtcvad_devel_hint()
            sys.exit(1)

    print("\n✅ Setup complete!")
    print("🔄 Restarting with virtualenv Python...\n")
    
    # Re-execute with virtualenv Python
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Check Python version, then setup virtualenv BEFORE importing dependencies
check_python_version()
setup_virtualenv()

# ============================================================================
# NOW SAFE TO IMPORT - We're running in virtualenv
# ============================================================================

import argparse
import collections
import wave
import io
import re
import tempfile
import textwrap
import time
from datetime import datetime
import select
import termios
import tty
from threading import Thread
import queue

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
    _mock.get_distribution = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = _mock

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import webrtcvad

import argostranslate.package
import argostranslate.translate
from faster_whisper import WhisperModel

# ============================================================================
# LANGUAGE CONSTANTS - single source of truth
# ============================================================================

LANG_LABELS = {"en": "English", "fr": "French", "es": "Spanish", "de": "German"}
LANG_FLAGS  = {"en": "🇬🇧",     "fr": "🇫🇷",     "es": "🇪🇸",     "de": "🇩🇪"}


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
            'w': 'Change Whisper model (tiny/base/small/medium/large)',
            'l': 'Change language (en/fr/es/de)',
            'i': 'Toggle Italian display',
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
        elif key == 'l':
            self._change_language()
        elif key == 'i':
            self._toggle_italian()
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
                self.translator.target_lang,
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
        """Cycle through Whisper models: tiny → base → small → medium → large-v3 → tiny"""
        models = ["tiny", "base", "small", "medium", "large-v3"]
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

    def _change_language(self):
        """Cycle through target languages: en → fr → es → de → en"""
        langs = list(LANG_LABELS.keys())
        current_idx = langs.index(self.translator.target_lang)
        new_lang = langs[(current_idx + 1) % len(langs)]

        print(f"\n🌐 Switching language to {LANG_LABELS[new_lang]}...", flush=True)

        # Install Argos model if needed (may take a moment)
        LiveTranslator._install_translation_model(new_lang)

        # Update translator
        self.translator.target_lang = new_lang
        self.translator.writer.target_flag = LANG_FLAGS.get(new_lang, "🇬🇧")

        print(f"✅ Language: Italian → {LANG_LABELS[new_lang]}", flush=True)

    def _toggle_italian(self):
        """Toggle Italian text display"""
        self.translator.show_italian = not self.translator.show_italian
        
        if self.translator.show_italian:
            print("\n🇮🇹 Italian display: ON", flush=True)
        else:
            print("\n🇮🇹 Italian display: OFF", flush=True)
    
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
        print(f"  Language : Italian -> {LANG_LABELS.get(t.target_lang, t.target_lang)}")
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
    """Handle audio stream capture using PipeWire/PulseAudio"""
    
    @staticmethod
    def _get_descriptions():
        """Return dict mapping source name -> human-readable description"""
        descriptions = {}
        try:
            env = {**os.environ, "LC_ALL": "C"}
            result = subprocess.run(
                ["pactl", "list", "sources"],
                capture_output=True, text=True, check=True, env=env
            )
            current_name = None
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("Name:"):
                    current_name = line.split(":", 1)[1].strip()
                elif line.startswith("Description:") and current_name:
                    desc = line.split(":", 1)[1].strip()
                    # Strip "Monitor of " prefix for cleaner display
                    if desc.startswith("Monitor of "):
                        desc = desc[len("Monitor of "):]
                    descriptions[current_name] = desc
                    current_name = None
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return descriptions

    @staticmethod
    def get_active_streams():
        """Return list of (id, name, description) for all active monitor sources"""
        try:
            env = {**os.environ, "LC_ALL": "C"}
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True, text=True, check=True, env=env
            )
            descriptions = AudioCapture._get_descriptions()
            streams = []
            for line in result.stdout.splitlines():
                if "monitor" in line and ("RUNNING" in line or "IDLE" in line):
                    parts = line.split()
                    source_id, source_name = parts[0], parts[1]
                    # Skip PipeWire internal loopback sinks
                    if source_name.startswith("input.loopback."):
                        continue
                    desc = descriptions.get(source_name, source_name)
                    streams.append((source_id, source_name, desc))
            return streams
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
    
    @staticmethod
    def capture_audio(source_name, segment_size):
        """Capture audio segment from stream"""
        cmd = [
            "parec",
            f"--device={source_name}",
            "--format=s16le",
            "--rate=16000",
            "--channels=1"
        ]
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            audio_data = proc.stdout.read(segment_size)
        finally:
            proc.terminate()
            proc.wait()

        return audio_data


class ModelConfig:
    """Whisper model configurations - optimized for Italian"""
    
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

    def __init__(self, filepath, model, mode, target_lang="en"):
        self.filepath = filepath
        self.file_handle = None
        self.target_flag = LANG_FLAGS.get(target_lang, "🇬🇧")

        if filepath:
            try:
                self.file_handle = open(filepath, 'w', encoding='utf-8')
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.file_handle.write(f"# Live Voice Translation\n\n")
                self.file_handle.write(f"**Date:** {now}  \n")
                self.file_handle.write(f"**Model:** {model}  \n")
                self.file_handle.write(f"**Mode:** {mode}  \n")
                self.file_handle.write(f"**Language:** Italian → {LANG_LABELS.get(target_lang, target_lang)}  \n\n")
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
            self.file_handle.write(f"🇮🇹 *{text_it}*\n\n")
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


class LiveTranslator:
    """Main translation engine with keyboard control"""
    
    def __init__(self, model_name, mode, save_file=None, enable_keyboard=True, show_italian=False, vad_filter=True, target_lang="en", use_gpu=False):
        self.model_name = model_name
        self.mode = mode
        self.config = ModelConfig.get_config(model_name, mode)
        self.writer = TranscriptWriter(save_file, model_name, mode, target_lang)
        self.model = None
        self.vad_filter = vad_filter
        self.target_lang = target_lang
        self.use_gpu = use_gpu

        # Keyboard control
        self.enable_keyboard = enable_keyboard
        self.keyboard_controller = None
        self.paused = False
        self.should_quit = False

        # Display options
        self.show_italian = show_italian
        
        # Audio threading
        self.audio_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        self._tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._tmp_wav.close()  # keep only the name, avoid dual open handle

        # Rolling context passed to Whisper as initial_prompt (last ~200 chars of Italian)
        self._whisper_context = ""

        # Sentence accumulation buffer: holds fragments until a complete sentence is ready
        self._sentence_buffer = ""
        self._sentence_buffer_time = None  # time when buffer was last updated

        # Recent translated phrases for dedup (last 5 outputs)
        self._recent_phrases: collections.deque = collections.deque(maxlen=5)

        # Recent Italian segments for dedup (last 5 inputs)
        self._recent_italian: collections.deque = collections.deque(maxlen=5)

        # Session statistics
        self.session_start = datetime.now()
        self.phrase_count = 0
        self.word_count = 0
        self.dropped_count = 0
        
    def setup(self):
        """Initialize Whisper model, translation, and keyboard"""
        # Ensure required translation models are installed
        self._install_translation_model(self.target_lang)
        
        # Load Whisper model
        print(f"⏳ Loading model {self.model_name}...", flush=True)
        if self.use_gpu:
            print("⚠️  GPU mode requested — may not work on AMD (ROCm not supported by CTranslate2)", flush=True)
            try:
                self.model = WhisperModel(self.model_name, device="cuda", compute_type="float16")
                print("✅ GPU (CUDA) loaded\n", flush=True)
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
    def _install_translation_model(target_lang="en"):
        """Ensure required Argos translation models are installed (it→en, and en→target if needed)"""
        pairs = [("it", "en")]
        if target_lang != "en":
            pairs.append(("en", target_lang))

        try:
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            installed_packages = argostranslate.package.get_installed_packages()

            for from_code, to_code in pairs:
                if any(p.from_code == from_code and p.to_code == to_code for p in installed_packages):
                    continue
                package_to_install = next(
                    filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages),
                    None
                )
                if package_to_install is None:
                    print(f"Warning: No Argos model found for {from_code}→{to_code}")
                    continue
                print(f"⏳ Installing translation model {from_code}→{to_code}...", flush=True)
                argostranslate.package.install_from_path(package_to_install.download())
                print(f"✅ {from_code}→{to_code} installed", flush=True)
        except Exception as e:
            print(f"Warning: Could not install translation model: {e}")
    
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

        cmd = [
            "parec",
            f"--device={source_name}",
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        ring_buffer = collections.deque(maxlen=PRE_PADDING_FRAMES)
        voiced_frames = []
        in_speech = False
        silence_count = 0

        try:
            while not self.should_quit:
                frame = proc.stdout.read(FRAME_BYTES)
                if not frame or len(frame) < FRAME_BYTES:
                    break

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
                self._translate_and_display(clean, skip_italian=True)
                self._sentence_buffer = ""
                self._sentence_buffer_time = None
        
        # Skip very short audio (< 300 ms)
        if len(audio_data) < 9600:
            return
        
        # Create WAV buffer
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
        wav_buffer.seek(0)
        
        # Save to temporary file (unique per instance)
        with open(self._tmp_wav.name, "wb") as f:
            f.write(wav_buffer.read())

        # Transcribe with Whisper
        # initial_prompt feeds the last ~200 chars of Italian to give Whisper
        # context across VAD chunks: better capitalisation, punctuation, continuity
        # word_timestamps=True lets us stream Italian words as they are decoded
        segments, _ = self.model.transcribe(
            self._tmp_wav.name,
            language="it",
            vad_filter=self.vad_filter,
            beam_size=self.config["beam"],
            condition_on_previous_text=False,
            initial_prompt=self._whisper_context or None,
            word_timestamps=True,
        )

        # Translate and display
        separator_printed = False
        chunk_italian = []
        for segment in segments:
            # Drop hallucinated segments: Whisper loops on silence/noise
            # no_speech_prob > 0.6 means Whisper is not confident speech was heard
            # compression_ratio > 2.4 means the output is suspiciously repetitive
            if getattr(segment, 'no_speech_prob', 0.0) > 0.6:
                continue
            if getattr(segment, 'compression_ratio', 1.0) > 2.4:
                continue
            text_it = segment.text.strip()
            if text_it and len(text_it) > 3:
                # Skip entire segment if identical to a recently seen one
                text_it_norm = text_it.lower()
                if text_it_norm in self._recent_italian:
                    continue
                self._recent_italian.append(text_it_norm)

                # Remove Whisper loop artefacts within the segment text
                text_it = self._clean_hallucination_loops(text_it)

                chunk_italian.append(text_it)
                if not separator_printed:
                    print("─" * 50, flush=True)
                    separator_printed = True

                # Stream Italian words one by one with 20ms micro-delay (typewriter effect)
                if self.show_italian and segment.words:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"\033[92m[{timestamp}]", end=" ", flush=True)
                    for word_info in segment.words:
                        print(word_info.word, end="", flush=True)
                        time.sleep(0.02)
                    print("\033[0m", flush=True)

                # Accumulate fragments and only translate complete sentences
                for part in re.split(r'(?<=[.!?])\s+', text_it):
                    part = part.strip()
                    if not part or len(part) <= 5:
                        continue
                    self._sentence_buffer = (self._sentence_buffer + " " + part).strip() if self._sentence_buffer else part
                    self._sentence_buffer_time = time.time()
                    # Flush when the accumulated buffer ends with terminal punctuation (not ellipsis)
                    ends_sentence = bool(re.search(r'[.!?]\s*$', self._sentence_buffer)) and not self._sentence_buffer.rstrip().endswith('...')
                    too_long = len(self._sentence_buffer) > 200
                    if ends_sentence or too_long:
                        clean = re.sub(r'\s*\.\.\.\s*', ' ', self._sentence_buffer).strip()
                        self._translate_and_display(clean, skip_italian=True)
                        self._sentence_buffer = ""
                        self._sentence_buffer_time = None

        # Update rolling context for next chunk (keep last ~200 chars)
        if chunk_italian:
            combined = " ".join(chunk_italian)
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

    def _translate_and_display(self, text_it, skip_italian=False):
        """Translate Italian text and display/save"""
        # Step 1: it → en (always needed)
        text_en = argostranslate.translate.translate(text_it, "it", "en")

        # Step 2: en → target (if not English)
        if self.target_lang == "en":
            text_target = text_en
        else:
            text_target = argostranslate.translate.translate(text_en, "en", self.target_lang)

        # Skip exact duplicate of any recently displayed phrase (Whisper loop guard)
        text_target_norm = text_target.strip().lower()
        if text_target_norm in self._recent_phrases:
            return
        self._recent_phrases.append(text_target_norm)

        # Get timestamp and flag
        timestamp = datetime.now().strftime('%H:%M:%S')
        flag = LANG_FLAGS.get(self.target_lang, "🇬🇧")

        # Display Italian if enabled and not already streamed word by word
        if self.show_italian and not skip_italian:
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
        
        try:
            while not self.should_quit:
                # Check keyboard input (fast, non-blocking)
                if self.keyboard_controller:
                    key = self.keyboard_controller.get_key(timeout=0.01)
                    if key:
                        self.keyboard_controller.handle_key(key)
                
                # Get audio from queue (non-blocking with timeout)
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    self.process_audio(audio_data)
                except queue.Empty:
                    # No audio ready yet, continue checking keyboard
                    pass
                    
        except KeyboardInterrupt:
            print("\n\nStopping translation.")
        finally:
            # Signal threads to stop
            self.should_quit = True

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

            # Remove temporary wav file
            try:
                os.unlink(self._tmp_wav.name)
            except OSError:
                pass

            # Close file (with session stats if saving)
            self.writer.close(
                duration_str=duration_str,
                phrase_count=self.phrase_count,
                word_count=self.word_count,
                dropped_count=self.dropped_count,
            )


def show_menu():
    """Interactive model selection menu"""
    print("═══════════════════════════════════════════")
    print("   Italian Live Translation")
    print("═══════════════════════════════════════════")
    print()
    print("Available models:")
    print()
    print("  1) tiny   - Ultra fast     (60% accuracy, ~1.5s, 1GB RAM)")
    print("  2) base   - Fast           (85% accuracy, ~4s, 1.5GB RAM)")
    print("  3) small  - Balanced       (90% accuracy, ~5s, 2GB RAM)")
    print("  4) medium - Recommended    (95% accuracy, ~8s, 5GB RAM) [DEFAULT]")
    print("  5) large  - Maximum        (98% accuracy, ~12s, 10GB RAM) ⚠️  High fan")
    print()
    
    choice = input("Your choice (1-5 or Enter for default): ").strip()
    
    models = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large-v3",
        "": "medium"
    }
    
    return models.get(choice, "medium")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Real-time Italian→English/French/Spanish/German audio translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive menu
  %(prog)s medium                    # Medium model
  %(prog)s medium --save             # Save to auto-generated file
  %(prog)s large --slow --save meeting.md
  %(prog)s medium --show-italian     # Display Italian + translation
  %(prog)s medium --to fr            # Translate to French
  %(prog)s medium --to es            # Translate to Spanish
  %(prog)s medium --to de            # Translate to German
  %(prog)s medium --no-vad           # Disable Voice Activity Detection

Audio source:
  If multiple audio streams are active (e.g. Zoom + YouTube), an
  interactive menu lets you pick which one to translate.

Keyboard shortcuts (during execution):
  p - Pause/Resume
  s - Save transcript now (creates file if needed)
  m - Change mode (fast/normal/slow)
  w - Change Whisper model (tiny/base/small/medium/large)
  l - Change target language (en/fr/es/de)
  i - Toggle Italian display
  q - Quit
  h - Show help
        """
    )
    
    parser.add_argument(
        "model",
        nargs="?",
        choices=["tiny", "base", "small", "medium", "large"],
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
        "--show-italian", "-i",
        action="store_true",
        help="Display Italian text (toggle with 'i' key)"
    )
    
    parser.add_argument(
        "--to",
        choices=["en", "fr", "es", "de"],
        default="en",
        metavar="LANG",
        help="Target language: en (default), fr, es, de"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU/CUDA acceleration (NVIDIA only, falls back to CPU on failure)"
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
    
    # Determine model
    if args.model:
        model = args.model if args.model != "large" else "large-v3"
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
    if args.show_italian:
        print(f"  Italian    : Displayed")
    lang_chain = f"Italian → English → {LANG_LABELS[args.to]}" if args.to != "en" else "Italian → English"
    print(f"  Language   : {lang_chain}")
    print(f"  Device     : {'GPU (CUDA) — experimental' if args.gpu else 'CPU'}")
    print(f"  VAD        : {'Disabled' if args.no_vad else 'Enabled (skips silence)'}")
    print()
    
    # Detect audio streams
    streams = AudioCapture.get_active_streams()
    if not streams:
        print("Error: No active audio stream detected", file=sys.stderr)
        print("Launch a YouTube video or video call with sound", file=sys.stderr)
        sys.exit(1)

    if len(streams) == 1:
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
        show_italian=args.show_italian,
        vad_filter=not args.no_vad,
        target_lang=args.to,
        use_gpu=args.gpu,
    )
    translator.setup()
    translator.run(stream_id)


if __name__ == "__main__":
    main()
