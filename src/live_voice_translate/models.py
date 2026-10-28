# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whisper model sizes and the fast/normal/slow listening modes."""

from typing import ClassVar, TypedDict


class ModeSettings(TypedDict):
    """Beam width and silence timing for one model and mode."""

    beam: int
    vad_silence_ms: int
    vad_aggressiveness: int
    max_speech_ms: int
    precision: str


WHISPER_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3-turbo",
    "large-v3",
)
MODES: tuple[str, ...] = ("fast", "normal", "slow")
MODEL_ALIASES: dict[str, str] = {
    "large": "large-v3",
    "turbo": "large-v3-turbo",
}


class ModelConfig:
    """Whisper settings tuned for live Chinese speech."""

    # vad_silence_ms: silence that ends an utterance
    # vad_aggressiveness: Silero threshold, 0 permissive through 3 aggressive
    # max_speech_ms: force-flush after this long even without silence
    # beam: Whisper beam search width
    CONFIGS: ClassVar[dict[str, dict[str, ModeSettings]]] = {
        "tiny": {
            "fast": {
                "beam": 1,
                "vad_silence_ms": 500,
                "vad_aggressiveness": 3,
                "max_speech_ms": 2000,
                "precision": "60%",
            },
            "normal": {
                "beam": 1,
                "vad_silence_ms": 900,
                "vad_aggressiveness": 2,
                "max_speech_ms": 4000,
                "precision": "60%",
            },
            "slow": {
                "beam": 1,
                "vad_silence_ms": 1200,
                "vad_aggressiveness": 1,
                "max_speech_ms": 8000,
                "precision": "60%",
            },
        },
        "base": {
            "fast": {
                "beam": 1,
                "vad_silence_ms": 500,
                "vad_aggressiveness": 3,
                "max_speech_ms": 2000,
                "precision": "85%",
            },
            "normal": {
                "beam": 2,
                "vad_silence_ms": 900,
                "vad_aggressiveness": 2,
                "max_speech_ms": 4000,
                "precision": "85%",
            },
            "slow": {
                "beam": 3,
                "vad_silence_ms": 1200,
                "vad_aggressiveness": 1,
                "max_speech_ms": 8000,
                "precision": "85%",
            },
        },
        "small": {
            "fast": {
                "beam": 1,
                "vad_silence_ms": 500,
                "vad_aggressiveness": 3,
                "max_speech_ms": 2000,
                "precision": "90%",
            },
            "normal": {
                "beam": 2,
                "vad_silence_ms": 900,
                "vad_aggressiveness": 2,
                "max_speech_ms": 4000,
                "precision": "90%",
            },
            "slow": {
                "beam": 3,
                "vad_silence_ms": 1200,
                "vad_aggressiveness": 1,
                "max_speech_ms": 8000,
                "precision": "90%",
            },
        },
        "medium": {
            "fast": {
                "beam": 2,
                "vad_silence_ms": 500,
                "vad_aggressiveness": 3,
                "max_speech_ms": 2000,
                "precision": "95%",
            },
            "normal": {
                "beam": 3,
                "vad_silence_ms": 900,
                "vad_aggressiveness": 2,
                "max_speech_ms": 4000,
                "precision": "95%",
            },
            "slow": {
                "beam": 4,
                "vad_silence_ms": 1200,
                "vad_aggressiveness": 1,
                "max_speech_ms": 8000,
                "precision": "95%",
            },
        },
        "large-v3-turbo": {
            "fast": {
                "beam": 1,
                "vad_silence_ms": 500,
                "vad_aggressiveness": 3,
                "max_speech_ms": 3000,
                "precision": "high",
            },
            "normal": {
                "beam": 2,
                "vad_silence_ms": 900,
                "vad_aggressiveness": 2,
                "max_speech_ms": 6000,
                "precision": "high",
            },
            "slow": {
                "beam": 3,
                "vad_silence_ms": 1200,
                "vad_aggressiveness": 1,
                "max_speech_ms": 10000,
                "precision": "high",
            },
        },
        "large-v3": {
            "fast": {
                "beam": 2,
                "vad_silence_ms": 500,
                "vad_aggressiveness": 3,
                "max_speech_ms": 2000,
                "precision": "98%",
            },
            "normal": {
                "beam": 3,
                "vad_silence_ms": 900,
                "vad_aggressiveness": 2,
                "max_speech_ms": 4000,
                "precision": "98%",
            },
            "slow": {
                "beam": 4,
                "vad_silence_ms": 1200,
                "vad_aggressiveness": 1,
                "max_speech_ms": 8000,
                "precision": "98%",
            },
        },
    }

    MODE_DESC: ClassVar[dict[str, str]] = {
        "fast": "500ms pause — reactive, may cut mid-sentence",
        "normal": "800ms pause — balanced",
        "slow": "1200ms pause — waits for complete sentences",
    }

    @classmethod
    def get_config(cls, model: str, mode: str = "normal") -> ModeSettings:
        """Return the settings for a Whisper model and listening mode."""
        return cls.CONFIGS.get(model, cls.CONFIGS["medium"])[mode]
