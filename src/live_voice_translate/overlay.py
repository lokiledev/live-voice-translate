# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Always-on-top window for the latest source sentence and its translation."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import glfw
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFont
from Xlib import X, display
from Xlib.protocol.event import ClientMessage

if TYPE_CHECKING:
    from glfw import _GLFWwindowPointerT

_WRAP_TOKEN_RE = re.compile(
    r"[\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]"
    r"|[^\s\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]+"
    r"|\s+"
)
_Font = ImageFont.ImageFont | ImageFont.FreeTypeFont


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: _Font, max_width: int) -> list[str]:
    """Wrap at spaces for Latin text and between characters for Chinese."""
    lines: list[str] = []
    current = ""
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


class Overlay:
    """Last sentence on top, translation below, on a half-transparent panel.

    Drag anywhere except the close mark. The window manager performs the move.
    """

    WIDTH = 920
    HEIGHT = 240
    # (path, face index). The face has to cover both CJK and Latin glyphs.
    _FONTS: tuple[tuple[str, int], ...] = (
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),  # Noto Sans CJK SC
        ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 2),
        ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc", 2),
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
    )
    _MAX_FONT_SIZE = 30
    _MIN_FONT_SIZE = 14
    _PADDING = 20

    def __init__(self) -> None:
        """Create an overlay that is not on screen until ``open``."""
        self._source = ""
        self._target = ""
        self._font_file = next(
            ((path, index) for path, index in self._FONTS if Path(path).is_file()),
            None,
        )
        self._closed = False
        self._window: _GLFWwindowPointerT | None = None
        self._texture: int | None = None
        self._texture_key: tuple[tuple[str, str], str, int, int] | None = None
        # on_key receives a shortcut pressed while the overlay is focused.
        # status returns the current mode name and whether capture is paused.
        self.on_key: Callable[[str], None] | None = None
        self.status: Callable[[], tuple[str, bool]] | None = None

    def open(self) -> None:
        """Create the transparent window. GLFW can only composite it through X11."""
        os.environ.pop("WAYLAND_DISPLAY", None)
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
        if not glfw.init():
            raise RuntimeError("GLFW failed to initialize")

        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
        self._window = glfw.create_window(
            self.WIDTH,
            self.HEIGHT,
            "live-voice-translate",
            None,
            None,
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

    def push(self, source: str, target: str) -> None:
        """Replace the sentence currently on screen."""
        source = " ".join(source.split())
        target = " ".join(target.split())
        if source or target:
            self._source, self._target = source, target

    def pump(self) -> bool:
        """Draw one frame. Return False once the window has been closed."""
        window = self._window
        if window is None or self._closed:
            self._closed = True
            return False
        glfw.poll_events()
        if glfw.window_should_close(window):
            self._closed = True
            return False
        self._draw()
        glfw.swap_buffers(window)
        return True

    def close(self) -> None:
        """Destroy the window if it was opened."""
        self._closed = True
        window = self._window
        if window is not None:
            glfw.destroy_window(window)
            glfw.terminate()
        self._window = None

    def _shown_text(self) -> tuple[str, str]:
        if not (self._source or self._target):
            return "Waiting for speech…", ""
        return self._source, self._target

    def _font(self, size: int) -> _Font:
        if self._font_file is None:
            return ImageFont.load_default(size)
        path, index = self._font_file
        return ImageFont.truetype(path, size, index=index)

    def _fit(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        max_width: int,
        max_height: int,
    ) -> tuple[_Font, list[str], int]:
        """Largest font whose wrapped text fits, plus its lines and line height."""
        font = self._font(self._MIN_FONT_SIZE)
        lines: list[str] = []
        line_height = int(self._MIN_FONT_SIZE * 1.35)
        for size in range(self._MAX_FONT_SIZE, self._MIN_FONT_SIZE - 1, -2):
            font = self._font(size)
            lines = _wrap_text(draw, text, font, max_width)
            line_height = int(size * 1.35)
            if len(lines) * line_height <= max_height:
                return font, lines, line_height
        max_lines = max(1, max_height // line_height)
        return font, lines[-max_lines:], line_height

    def _on_key(
        self,
        _window: _GLFWwindowPointerT,
        key: int,
        scancode: int,
        action: int,
        _mods: int,
    ) -> None:
        """Forward letter shortcuts typed into the overlay."""
        if action != glfw.PRESS or self.on_key is None:
            return
        name = glfw.get_key_name(key, scancode)
        if name and len(name) == 1 and name.isalpha():
            self.on_key(name.lower())

    def _status_line(self) -> str:
        if self.status is None:
            return ""
        mode, paused = self.status()
        state = "PAUSED" if paused else f"mode: {mode}"
        return f"{state}    M mode   P pause   Q quit"

    def _on_mouse(
        self,
        window: _GLFWwindowPointerT,
        button: int,
        action: int,
        _mods: int,
    ) -> None:
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        width, _height = glfw.get_window_size(window)
        x_pos, y_pos = glfw.get_cursor_pos(window)
        if x_pos >= width - 42 and y_pos <= 42:
            glfw.set_window_should_close(window, True)
            return
        self._begin_move(x_pos, y_pos)

    def _begin_move(self, x_pos: float, y_pos: float) -> None:
        """Ask the window manager to drag the window from this click."""
        window = self._window
        if window is None:
            return
        xid = glfw.get_x11_window(window)
        if not xid:
            return
        origin_x, origin_y = glfw.get_window_pos(window)
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

    def _draw(self) -> None:
        window = self._window
        if window is None:
            return
        width, height = glfw.get_framebuffer_size(window)
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.0, 0.0, 0.0, 0.5)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self._draw_text(width, height)

    def _draw_text(self, width: int, height: int) -> None:
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
        GL.glTexCoord2f(0, 0)
        GL.glVertex2f(0, 0)
        GL.glTexCoord2f(1, 0)
        GL.glVertex2f(width, 0)
        GL.glTexCoord2f(1, 1)
        GL.glVertex2f(width, height)
        GL.glTexCoord2f(0, 1)
        GL.glVertex2f(0, height)
        GL.glEnd()

    def _upload_text(self, width: int, height: int) -> None:
        image = self._render(width, height)
        if self._texture is None:
            self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            image.tobytes(),
        )

    def _render(self, width: int, height: int) -> Image.Image:
        """Draw the text panel into a transparent RGBA image."""
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        pad = self._PADDING
        status = self._status_line()
        footer_h = 30 if status else 0
        half = (height - footer_h) // 2
        text_width = width - 2 * pad - 30
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
