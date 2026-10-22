# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Load native libraries that need setup before their first import.

CTranslate2's wheels look for the CUDA 12 cuBLAS sonames, and webrtcvad still
imports ``pkg_resources``. Both are arranged here, then the libraries are
imported so the rest of the package can use them.
"""

import ctypes
import glob
import importlib.util
import io
import sys
import types
import warnings


def install() -> None:
    """Silence warnings, mock pkg_resources, and preload CUDA 12 cuBLAS."""
    warnings.filterwarnings("ignore")
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    _mock_pkg_resources()
    _preload_cuda12_cublas()


class _PkgResources(types.ModuleType):
    """Stand-in for the pkg_resources module removed in setuptools 81."""

    def get_distribution(self, *_args: object, **_kwargs: object) -> types.SimpleNamespace:
        """Return a distribution object with the version attribute webrtcvad reads."""
        return types.SimpleNamespace(version="0")


def _mock_pkg_resources() -> None:
    """Install a pkg_resources stand-in when setuptools no longer provides one."""
    if importlib.util.find_spec("pkg_resources") is not None:
        return
    sys.modules["pkg_resources"] = _PkgResources("pkg_resources")


def _preload_cuda12_cublas() -> None:
    """Make the pip-installed CUDA 12 cuBLAS visible to CTranslate2."""
    try:
        import nvidia.cublas  # noqa: PLC0415
    except ImportError:
        return
    for path in nvidia.cublas.__path__:
        for lib in ("libcublasLt.so.12", "libcublas.so.12"):
            for match in glob.glob(f"{path}/lib/{lib}"):
                ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)


install()

import ctranslate2
import webrtcvad
from faster_whisper import WhisperModel

__all__ = ["WhisperModel", "ctranslate2", "webrtcvad"]
