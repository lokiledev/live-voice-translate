# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Session loop: capture speech, transcribe it, and translate complete sentences."""

from __future__ import annotations

import collections
import queue
import re
import subprocess
import textwrap
import time
from datetime import datetime
from threading import Thread
from typing import TYPE_CHECKING

import argostranslate.package
import argostranslate.translate
import numpy as np

from live_voice_translate.audio import AudioCapture
from live_voice_translate.cuda import cuda_device_name
from live_voice_translate.keyboard import KeyboardController
from live_voice_translate.languages import SOURCE_LANG, TARGET_LANG
from live_voice_translate.models import ModelConfig, ModeSettings
from live_voice_translate.native import WhisperModel, webrtcvad
from live_voice_translate.text import clean_hallucination_loops, is_chinese
from live_voice_translate.transcript import TranscriptWriter

if TYPE_CHECKING:
    from live_voice_translate.overlay import Overlay


class LiveTranslator:
    """Transcribe live audio and print the English as sentences complete."""

    def __init__(
        self,
        model_name: str,
        mode: str,
        save_file: str | None = None,
        *,
        enable_keyboard: bool = True,
        show_chinese: bool = False,
        vad_filter: bool = True,
        use_gpu: bool = False,
        mixed: bool = False,
        overlay: bool = False,
    ) -> None:
        """Prepare a session. Models are loaded in ``setup``."""
        self.model_name = model_name
        self.mode = mode
        self.mixed = mixed
        self.overlay: Overlay | None = None
        if overlay:
            from live_voice_translate.overlay import Overlay as OverlayWindow  # noqa: PLC0415

            self.overlay = OverlayWindow()
        self.config: ModeSettings = ModelConfig.get_config(model_name, mode)
        self.writer = TranscriptWriter(save_file, model_name, mode)
        self.model: WhisperModel | None = None
        self.vad_filter = vad_filter
        self.use_gpu = use_gpu

        self.enable_keyboard = enable_keyboard
        self.keyboard_controller: KeyboardController | None = None
        self.paused = False
        self.should_quit = False
        self.show_chinese = show_chinese

        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self.capture_thread: Thread | None = None
        self._whisper_context = ""
        self._sentence_buffer = ""
        self._sentence_buffer_time: float | None = None
        self._recent_phrases: collections.deque[str] = collections.deque(maxlen=5)
        self._recent_chinese: collections.deque[str] = collections.deque(maxlen=5)

        self.session_start = datetime.now()
        self.phrase_count = 0
        self.word_count = 0
        self.dropped_count = 0

    def setup(self) -> None:
        """Load the Argos pair, the Whisper model, and the terminal keys."""
        self._install_translation_model()

        print(f"⏳ Loading model {self.model_name}...", flush=True)
        if self.use_gpu:
            print(f"⏳ Loading on GPU ({cuda_device_name()})...", flush=True)
            try:
                self.model = WhisperModel(self.model_name, device="cuda", compute_type="float16")
                print("✅ GPU loaded\n", flush=True)
            except Exception as exc:
                print(f"⚠️  GPU failed ({exc}), falling back to CPU", flush=True)
                self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                self.use_gpu = False
        else:
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        print("✅ Ready\n", flush=True)

        if self.enable_keyboard:
            self.keyboard_controller = KeyboardController(self)
            if not self.keyboard_controller.setup():
                self.keyboard_controller = None

    @staticmethod
    def _install_translation_model() -> None:
        """Ensure the direct Chinese to English Argos model is installed."""
        try:
            installed_packages = argostranslate.package.get_installed_packages()
            if any(
                package.from_code == SOURCE_LANG and package.to_code == TARGET_LANG
                for package in installed_packages
            ):
                return

            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                (
                    package
                    for package in available_packages
                    if package.from_code == SOURCE_LANG and package.to_code == TARGET_LANG
                ),
                None,
            )
            if package_to_install is None:
                raise RuntimeError(f"No Argos model found for {SOURCE_LANG}→{TARGET_LANG}")

            print(f"⏳ Installing translation model {SOURCE_LANG}→{TARGET_LANG}...", flush=True)
            argostranslate.package.install_from_path(package_to_install.download())
            print(f"✅ {SOURCE_LANG}→{TARGET_LANG} installed", flush=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not install the Chinese→English translation model: {exc}"
            ) from exc

    def _audio_capture_loop(self, source_name: str) -> None:
        """Read the monitor continuously and queue utterances split by silence."""
        sample_rate = 16000
        frame_ms = 30
        frame_bytes = sample_rate * frame_ms // 1000 * 2
        pre_padding_frames = 300 // frame_ms
        min_speech_frames = 500 // frame_ms

        current_aggressiveness = self.config["vad_aggressiveness"]
        vad = webrtcvad.Vad(current_aggressiveness)

        proc: subprocess.Popen[bytes] | None = None
        ring_buffer: collections.deque[bytes] = collections.deque(maxlen=pre_padding_frames)
        voiced_frames: list[bytes] = []
        in_speech = False
        silence_count = 0

        def start_recorder() -> subprocess.Popen[bytes]:
            return subprocess.Popen(
                AudioCapture.record_command(source_name),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

        def read_frame(process: subprocess.Popen[bytes]) -> bytes:
            stdout = process.stdout
            if stdout is None:
                return b""
            return stdout.read(frame_bytes)

        try:
            proc = start_recorder()
            while not self.should_quit:
                frame = read_frame(proc)
                if not frame or len(frame) < frame_bytes:
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

                new_aggressiveness = self.config["vad_aggressiveness"]
                if new_aggressiveness != current_aggressiveness:
                    current_aggressiveness = new_aggressiveness
                    vad = webrtcvad.Vad(current_aggressiveness)
                silence_frames = self.config["vad_silence_ms"] // frame_ms

                try:
                    is_speech = vad.is_speech(frame, sample_rate)
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
                            if len(voiced_frames) >= min_speech_frames:
                                self._enqueue_audio(b"".join(voiced_frames))
                            in_speech = False
                            voiced_frames = []
                            ring_buffer.clear()
                            silence_count = 0
                    else:
                        silence_count = 0
                        max_speech_frames = self.config["max_speech_ms"] // frame_ms
                        if len(voiced_frames) >= max_speech_frames:
                            self._enqueue_audio(b"".join(voiced_frames))
                            overlap_frames = 500 // frame_ms
                            voiced_frames = list(voiced_frames[-overlap_frames:])
        except Exception as exc:
            if not self.should_quit:
                print(f"\n⚠️  Audio capture error: {exc}", flush=True)
        finally:
            if proc is not None:
                proc.terminate()
                proc.wait()

    def _enqueue_audio(self, audio_data: bytes) -> None:
        try:
            self.audio_queue.put(audio_data, timeout=0.5)
        except queue.Full:
            self.dropped_count += 1

    def process_audio(self, audio_data: bytes) -> None:
        """Transcribe one utterance and translate any complete sentences in it."""
        if self.paused:
            return

        if self._sentence_buffer and self._sentence_buffer_time is not None:
            if time.time() - self._sentence_buffer_time > 1.5:
                clean = re.sub(r"\s*\.\.\.\s*", " ", self._sentence_buffer).strip()
                self._translate_and_display(clean, skip_source=True)
                self._sentence_buffer = ""
                self._sentence_buffer_time = None

        if len(audio_data) < 9600 or self.model is None:
            return

        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            audio,
            language=None if self.mixed else SOURCE_LANG,
            multilingual=self.mixed,
            vad_filter=self.vad_filter,
            beam_size=self.config["beam"],
            condition_on_previous_text=False,
            initial_prompt=self._whisper_context or None,
            word_timestamps=True,
        )

        separator_printed = False
        chunk_chinese: list[str] = []
        for segment in segments:
            if getattr(segment, "no_speech_prob", 0.0) > 0.6:
                continue
            if getattr(segment, "compression_ratio", 1.0) > 2.4:
                continue
            text_it = segment.text.strip()
            if not text_it or len(text_it) < 2:
                continue
            text_it_norm = text_it.lower()
            if text_it_norm in self._recent_chinese:
                continue
            self._recent_chinese.append(text_it_norm)

            text_it = clean_hallucination_loops(text_it)
            chunk_chinese.append(text_it)
            if not separator_printed:
                print("─" * 50, flush=True)
                separator_printed = True

            words = getattr(segment, "words", None)
            if self.show_chinese and words:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\033[92m[{timestamp}]", end=" ", flush=True)
                for word_info in words:
                    print(word_info.word, end="", flush=True)
                    time.sleep(0.02)
                print("\033[0m", flush=True)

            for part in re.split(r"(?<=[.!?。！？])\s*", text_it):
                part = part.strip()
                if not part or len(part) < 2:
                    continue
                if (
                    self.mixed
                    and self._sentence_buffer
                    and is_chinese(part) != is_chinese(self._sentence_buffer)
                ):
                    self._translate_and_display(self._sentence_buffer, skip_source=True)
                    self._sentence_buffer = ""
                    self._sentence_buffer_time = None
                self._sentence_buffer = (
                    f"{self._sentence_buffer} {part}".strip() if self._sentence_buffer else part
                )
                self._sentence_buffer_time = time.time()
                ends_sentence = bool(
                    re.search(r"[.!?。！？]\s*$", self._sentence_buffer)
                ) and not self._sentence_buffer.rstrip().endswith("...")
                too_long = len(self._sentence_buffer) > 200
                if ends_sentence or too_long:
                    clean = re.sub(r"\s*\.\.\.\s*", " ", self._sentence_buffer).strip()
                    self._translate_and_display(clean, skip_source=True)
                    self._sentence_buffer = ""
                    self._sentence_buffer_time = None

        if chunk_chinese:
            combined = " ".join(chunk_chinese)
            self._whisper_context = (self._whisper_context + " " + combined)[-200:].lstrip()

    def _translate_and_display(self, text_it: str, *, skip_source: bool = False) -> None:
        """Translate one Chinese sentence and print, save, and overlay it."""
        if self.mixed and not is_chinese(text_it):
            text_target = text_it
        else:
            text_target = argostranslate.translate.translate(text_it, SOURCE_LANG, TARGET_LANG)

        text_target_norm = text_target.strip().lower()
        if text_target_norm in self._recent_phrases:
            return
        self._recent_phrases.append(text_target_norm)
        if self.overlay is not None:
            self.overlay.push(text_it, text_target)

        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.show_chinese and not skip_source:
            if len(text_it) > 70:
                output_it = textwrap.fill(
                    text_it,
                    width=70,
                    initial_indent=f"[{timestamp}] ",
                    subsequent_indent=" " * 13,
                )
            else:
                output_it = f"[{timestamp}] {text_it}"
            print(f"\033[92m{output_it}\033[0m", flush=True)

        words = text_target.split()
        print(f"[{timestamp}] ▶ ", end="", flush=True)
        for word in words:
            print(word + " ", end="", flush=True)
            time.sleep(0.02)
        print(flush=True)

        self.phrase_count += 1
        self.word_count += len(text_target.split())
        self.writer.write(timestamp, text_it, text_target)

    def run(self, stream_id: str) -> None:
        """Capture ``stream_id`` until quit, a closed overlay, or Ctrl+C."""
        self.capture_thread = Thread(
            target=self._audio_capture_loop,
            args=(stream_id,),
            daemon=True,
        )
        self.capture_thread.start()

        if self.overlay is not None:
            try:
                controller = self.keyboard_controller or KeyboardController(self)
                self.overlay.on_key = controller.handle_key
                self.overlay.status = lambda: (self.mode, self.paused)
                self.overlay.open()
            except Exception as exc:
                print(f"⚠️  Overlay failed to open: {exc}", flush=True)
                self.overlay.close()
                self.overlay = None

        try:
            while not self.should_quit:
                if self.overlay is not None and not self.overlay.pump():
                    self.should_quit = True
                    break

                if self.keyboard_controller is not None:
                    key = self.keyboard_controller.get_key(timeout=0.01)
                    if key:
                        self.keyboard_controller.handle_key(key)

                try:
                    audio_data = self.audio_queue.get(timeout=0.02 if self.overlay else 0.1)
                    self.process_audio(audio_data)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            print("\n\nStopping translation.")
        finally:
            self.should_quit = True
            if self.overlay is not None:
                self.overlay.close()
            if self.keyboard_controller is not None:
                self.keyboard_controller.cleanup()
            if self.capture_thread is not None and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)

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
                print("  Dropped    : 0")
            print("═══════════════════════════════════════════")
            print()

            self.writer.close(
                duration_str=duration_str,
                phrase_count=self.phrase_count,
                word_count=self.word_count,
                dropped_count=self.dropped_count,
            )
