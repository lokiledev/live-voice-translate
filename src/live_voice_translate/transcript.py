# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Markdown transcript files."""

from datetime import datetime
from typing import TextIO

from live_voice_translate.languages import SOURCE_FLAG, SOURCE_LABEL, TARGET_FLAG, TARGET_LABEL


class TranscriptWriter:
    """Save transcripts to a Markdown file."""

    def __init__(self, filepath: str | None, model: str, mode: str) -> None:
        """Open the transcript file and write its header."""
        self.filepath = filepath
        self.file_handle: TextIO | None = None
        self.target_flag = TARGET_FLAG

        if not filepath:
            return
        try:
            self.file_handle = open(filepath, "w", encoding="utf-8")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.file_handle.write("# Live Voice Translation\n\n")
            self.file_handle.write(f"**Date:** {now}  \n")
            self.file_handle.write(f"**Model:** {model}  \n")
            self.file_handle.write(f"**Mode:** {mode}  \n")
            self.file_handle.write(f"**Language:** {SOURCE_LABEL} → {TARGET_LABEL}  \n\n")
            self.file_handle.write("---\n\n")
            self.file_handle.flush()
            print(f"💾 Saving to: {filepath}\n", flush=True)
        except Exception as exc:
            print(f"⚠️  Warning: Could not open save file: {exc}\n", flush=True)
            self.file_handle = None

    def write(self, timestamp: str, text_it: str, text_target: str) -> None:
        """Append one translated sentence."""
        if self.file_handle is None:
            return
        self.file_handle.write(f"**[{timestamp}]**\n\n")
        self.file_handle.write(f"{SOURCE_FLAG} *{text_it}*\n\n")
        self.file_handle.write(f"{self.target_flag} {text_target}\n\n")
        self.file_handle.write("---\n\n")
        self.file_handle.flush()

    def close(
        self,
        duration_str: str | None = None,
        phrase_count: int | None = None,
        word_count: int | None = None,
        dropped_count: int | None = None,
    ) -> None:
        """Close the file, appending the session summary when one was collected."""
        if self.file_handle is None:
            return
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file_handle.write("\n---\n\n")
        self.file_handle.write(f"**End of session:** {end_time}  \n")
        if duration_str is not None:
            self.file_handle.write(f"**Duration:** {duration_str}  \n")
            self.file_handle.write(f"**Phrases:** {phrase_count}  \n")
            self.file_handle.write(f"**Words:** {word_count}  \n")
            if dropped_count:
                self.file_handle.write(
                    f"**Dropped:** {dropped_count} ⚠️ CPU too slow for this model  \n"
                )
        self.file_handle.close()
        print(f"\n💾 Transcript saved to: {self.filepath}")
