# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""NVIDIA device detection for Whisper."""

import subprocess

from live_voice_translate.native import ctranslate2


def cuda_available() -> bool:
    """Return whether CTranslate2 can see an NVIDIA GPU."""
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def cuda_device_name() -> str:
    """Return the first NVIDIA device name, or a generic CUDA label."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        name = result.stdout.strip().splitlines()[0].strip()
        return name or "CUDA"
    except Exception:
        return "CUDA"
