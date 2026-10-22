# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Text checks shared by transcription and translation."""

import re

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


def is_chinese(text: str) -> bool:
    """Whether text is mainly Chinese.

    One Chinese character carries roughly one word, so characters are compared
    against Latin words: "这个 feature 我们下周 release" counts as Chinese.
    """
    return len(_CJK_RE.findall(text)) >= max(1, len(_LATIN_WORD_RE.findall(text)))


def clean_hallucination_loops(text: str) -> str:
    """Collapse a clause that Whisper repeated three or more times.

    Two repetitions are kept, because they may be intentional emphasis.
    """
    return re.sub(
        r"(\b\w[\w'\s]{2,40}?)(?:,\s*\1){2,}",
        r"\1",
        text,
        flags=re.IGNORECASE,
    ).strip()
