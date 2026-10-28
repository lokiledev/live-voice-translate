# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Streaming speech detection with Silero VAD v6 on CPU.

The bundled ONNX file is the official v6.2.3 export. Its inputs are ``input``,
``state``, and ``sr``. See ``assets/LICENSE.silero-vad`` for the model license.
"""

from importlib.resources import files

import numpy as np
import onnxruntime

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512
FRAME_MS = WINDOW_SAMPLES * 1000 // SAMPLE_RATE
CONTEXT_SAMPLES = 64

# Former webrtcvad levels, 0 permissive through 3 aggressive.
_THRESHOLDS = (0.25, 0.35, 0.5, 0.65)
_MODEL = files("live_voice_translate").joinpath("assets", "silero_vad.onnx")


def frames_for(duration_ms: int) -> int:
    """Return how many Silero windows cover at least ``duration_ms``."""
    return max(1, (duration_ms + FRAME_MS - 1) // FRAME_MS)


class SileroVad:
    """Speech probability for one 16 kHz stream. The session stays on CPU."""

    def __init__(self, aggressiveness: int = 2) -> None:
        """Load the bundled model and start from empty recurrent state."""
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            _MODEL.read_bytes(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._threshold = 0.5
        self.set_aggressiveness(aggressiveness)

    def set_aggressiveness(self, level: int) -> None:
        """Map a 0–3 level onto the probability required to enter speech."""
        if level not in range(len(_THRESHOLDS)):
            msg = f"aggressiveness must be 0..{len(_THRESHOLDS) - 1}"
            raise ValueError(msg)
        self._threshold = _THRESHOLDS[level]

    def reset(self) -> None:
        """Drop recurrent state after a gap in the captured audio."""
        self._state.fill(0)
        self._context.fill(0)

    def is_speech(self, frame: bytes, *, in_speech: bool) -> bool:
        """Classify one 512-sample little-endian int16 frame.

        Once speech has started, the cutoff drops by 0.15 so a short dip does
        not flip the frame back to silence. The caller still waits out the
        mode's silence duration before ending the utterance.
        """
        expected = WINDOW_SAMPLES * 2
        if len(frame) != expected:
            msg = f"expected {expected} bytes, got {len(frame)}"
            raise ValueError(msg)
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        model_input = np.concatenate([self._context, samples.reshape(1, WINDOW_SAMPLES)], axis=1)
        probability, state = self._session.run(
            None,
            {"input": model_input, "state": self._state, "sr": self._sample_rate},
        )
        self._state = np.asarray(state, dtype=np.float32)
        self._context = model_input[:, -CONTEXT_SAMPLES:]
        cutoff = self._threshold if not in_speech else max(self._threshold - 0.15, 0.01)
        return float(np.asarray(probability).reshape(-1)[0]) >= cutoff
