# Copyright (C) 2026 Stéphane de Labrusse
# SPDX-License-Identifier: GPL-3.0-or-later
"""Find the PipeWire output that is playing audio and record its monitor."""

import json
import subprocess


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    return []


class AudioCapture:
    """Capture playback from PipeWire by monitoring the output device."""

    @staticmethod
    def record_command(target: str) -> list[str]:
        """Record a sink's monitor as 16 kHz mono PCM on stdout."""
        return [
            "pw-record",
            "--target",
            target,
            "--properties",
            "stream.capture.sink=true",
            "--rate",
            "16000",
            "--channels",
            "1",
            "--format",
            "s16",
            "--raw",
            "-",
        ]

    @staticmethod
    def _pw_dump() -> list[dict[str, object]]:
        try:
            result = subprocess.run(
                ["pw-dump"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
        try:
            parsed: object = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [_as_dict(item) for item in parsed if isinstance(item, dict)]

    @staticmethod
    def get_default_sink() -> tuple[str, str, str] | None:
        """Return (id, node name, description) of the default output device, or None."""
        objects = AudioCapture._pw_dump()
        default_name: str | None = None
        for obj in objects:
            if obj.get("type") != "PipeWire:Interface:Metadata":
                continue
            for entry_obj in _as_list(obj.get("metadata")):
                entry = _as_dict(entry_obj)
                if entry.get("key") != "default.audio.sink":
                    continue
                name_value = _as_dict(entry.get("value")).get("name")
                default_name = name_value if isinstance(name_value, str) else None
        for obj in objects:
            if obj.get("type") != "PipeWire:Interface:Node":
                continue
            props = _as_dict(_as_dict(obj.get("info")).get("props"))
            if props.get("media.class") != "Audio/Sink":
                continue
            name = props.get("node.name")
            node_name = name if isinstance(name, str) else ""
            if default_name is None or node_name == default_name:
                description = props.get("node.description")
                label = description if isinstance(description, str) and description else node_name
                return str(obj["id"]), node_name, label
        return None

    @staticmethod
    def get_active_streams() -> list[tuple[str, str, str]]:
        """Return (id, node name, description) for sinks that are playing audio."""
        objects = AudioCapture._pw_dump()

        nodes: dict[object, dict[str, str]] = {}
        for obj in objects:
            if obj.get("type") != "PipeWire:Interface:Node":
                continue
            info = _as_dict(obj.get("info"))
            props = _as_dict(info.get("props"))
            node_name = props.get("node.name")
            description = props.get("node.description")
            app = props.get("application.name")
            media = props.get("media.class")
            state = info.get("state")
            name = node_name if isinstance(node_name, str) else ""
            nodes[obj["id"]] = {
                "name": name,
                "description": description
                if isinstance(description, str) and description
                else name,
                "app": app if isinstance(app, str) and app else name,
                "media": media if isinstance(media, str) else "",
                "state": state if isinstance(state, str) else "",
            }

        apps_by_sink: dict[object, set[str]] = {}
        for obj in objects:
            if obj.get("type") != "PipeWire:Interface:Link":
                continue
            info = _as_dict(obj.get("info"))
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
            apps_by_sink.setdefault(info.get("input-node-id"), set()).add(source["app"])

        streams: list[tuple[str, str, str]] = []
        for sink_id, apps in apps_by_sink.items():
            sink = nodes[sink_id]
            if not sink["name"] or sink["name"].startswith("input.loopback."):
                continue
            desc = f"{sink['description']} ({', '.join(sorted(apps))})"
            streams.append((str(sink_id), sink["name"], desc))
        return streams

    @staticmethod
    def capture_audio(source_name: str, segment_size: int) -> bytes:
        """Capture one audio segment from a playing output device."""
        proc = subprocess.Popen(
            AudioCapture.record_command(source_name),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdout = proc.stdout
        try:
            if stdout is None:
                return b""
            return stdout.read(segment_size)
        finally:
            proc.terminate()
            proc.wait()
