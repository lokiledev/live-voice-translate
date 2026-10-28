# live-voice-translate

Real-time Chinese→English audio translation using Whisper and Argos Translate.

Captures system audio (YouTube, video calls, etc.) and provides instant transcription and translation with interactive keyboard controls.

## 🇨🇳 → 🇬🇧 Chinese edition

This fork is configured specifically for live Mandarin Chinese to English
translation. It captures desktop audio, transcribes Mandarin with a multilingual
Whisper model, then translates the Chinese transcript with Argos Translate's
offline `zh→en` model.

The recommended model is `large-v3-turbo`, downloaded automatically from
[Hugging Face](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo).
It offers a better real-time speed/accuracy tradeoff than `large-v3`.

<details>
<summary>Original project story</summary>

As a French developer working remotely with [Nethesis](https://www.nethesis.it/) (an Italian company behind NethServer and NethSecurity), I found myself in daily Italian-language meetings and calls. Not speaking Italian fluently, I needed a practical solution to follow technical discussions in real-time.

I built this tool initially for my own needs — a simple Python script to capture audio streams and translate them live during meetings. It quickly became indispensable for my daily work, allowing me to actively participate in Italian-speaking technical sessions, understand documentation being discussed, and keep up with fast-paced conversations.

After months of refining it for personal use, I realized this could help other developers in similar situations. Whether you're:
- 🤝 Working with Italian teams or clients
- 📚 Learning from Italian technical content (YouTube, conferences, webinars)
- 🌍 Contributing to Italian open-source projects
- 🎓 Studying Italian tech tutorials

...this tool can make your life significantly easier.

It demonstrates that complex problems don't always need complex solutions —
sometimes a straightforward Python script is all you need.

**Buon lavoro!** 🚀

</details>

## Features

- 🎤 **Real-time translation** - Mandarin Chinese to English, phrase by phrase
- 🎯 **6 Whisper models** - From tiny (fast) to large-v3 (accurate), including recommended large-v3-turbo
- ⚡ **3 speed modes** - Fast (500ms)/Normal (800ms)/Slow (1200ms) silence thresholds, switchable on-the-fly with **M**
- 🔇 **VAD-based chunking** - [webrtcvad](https://github.com/wiseman/py-webrtcvad) detects speech boundaries in real-time; flushes on silence rather than fixed intervals
- 📊 **Session statistics** - Duration, segments, word count and dropped chunks at end of session
- ⌨️ **Full keyboard control** - Pause, save, switch model, and change mode on-the-fly
- 💾 **Markdown export** - Save timestamped bilingual transcripts with session stats
- 🔧 **uv project** - Dependencies are declared in `pyproject.toml` and installed with `uv sync`
- ⚡ **GPU acceleration** - Uses an NVIDIA GPU automatically when CUDA is available; `--cpu` forces the CPU
- 🐧 **Linux native** - Captures PipeWire playback, including Chrome
- 🎛️ **Smart audio source selection** - Auto-detects active stream, interactive menu when multiple sources are active
- 🇨🇳 **Bilingual display** - Toggle Chinese source text visibility

## Privacy

All processing happens **entirely on your machine** — no audio, transcription, or translation data ever leaves your computer.

- **Speech recognition** is performed by [faster-whisper](https://github.com/guillaumekleindienst/faster-whisper), a local Whisper model that runs offline after the initial download.
- **Translation** is handled by [argostranslate](https://github.com/argosopentech/argos-translate), which uses locally installed language models with no network calls at runtime.
- **No cloud API** is contacted during use. There is no telemetry, no account, and no data sent to any third party.
- **Audio capture** reads your system audio stream in memory only; nothing is written to disk unless you explicitly use `--save`.

This tool is safe to use in environments where confidentiality matters (internal meetings, proprietary content, etc.).

## Requirements

- **OS**: Linux (tested on Fedora 43, Ubuntu 24.04, openSUSE Tumbleweed)
- **Python**: 3.12+ (this checkout is pinned to the system Python, currently 3.14)
- **Audio**: PipeWire (`pw-dump` and `pw-record`, included with PipeWire)

**Note**: The translator listens to the output device that is actually playing audio, so a Chrome tab does not need a separate monitor source.

## Installation

### 1. System dependencies

**Fedora / RHEL / CentOS:**
```bash
sudo dnf install pipewire
```

**openSUSE:**
```bash
sudo zypper install pipewire
```

**Ubuntu / Debian:**
```bash
sudo apt install pipewire
```

> PipeWire provides `pw-dump` and `pw-record`, which capture the audio an application is playing.

### 2. Clone and run

```bash
# Clone repository
git clone https://github.com/stephdl/live-voice-translate.git
cd live-voice-translate

# Create .venv and install the dependencies from pyproject.toml
uv sync

# Run
uv run lvt
```

`uv sync` installs the application and the `dev` dependency group (Ruff, ty, and pre-commit) into `.venv`. Use `uv sync --no-dev` to install only what `lvt` needs to run.

### 3. Updating

```bash
cd live-voice-translate
git pull origin main
uv sync
```

### 4. Development

The translator is the `live_voice_translate` package under `src/`. `lvt` is a thin console script. The dev tools are the `dev` dependency group, and pre-commit runs them:

```bash
uv run pre-commit install   # once per clone
uv run ruff check
uv run ruff format
uv run ty check
```

## Usage

### Default model

`lvt` loads `turbo` (`large-v3-turbo`) unless you pass another model name.

```bash
uv run lvt
uv run lvt --overlay
```

### Command-line
```bash
# Turbo model (recommended)
uv run lvt turbo

# Large model with slow mode (best quality)
uv run lvt large --slow

# Tiny model with fast mode (lowest latency)
uv run lvt tiny --fast

# Save transcript to file
uv run lvt medium --save meeting.md

# Auto-generated filename
uv run lvt medium --save

# Display Chinese + English
uv run lvt turbo --show-chinese

# Mixed Chinese/English speech: Chinese is translated, English is shown as is
uv run lvt turbo --mixed --show-chinese

# Floating half-transparent window: last sentence on top, its translation below
uv run lvt turbo --overlay

# Disable Voice Activity Detection (transcribe everything including silence)
uv run lvt medium --no-vad

# GPU is used automatically on NVIDIA. Force the CPU with:
uv run lvt medium --cpu
```

### Keyboard shortcuts

**During execution**, press:

| Key | Action |
|-----|--------|
| **P** | Pause/Resume translation |
| **S** | Save transcript now (creates file if needed) |
| **M** | Change mode (fast → normal → slow → fast) |
| **W** | Change Whisper model (tiny → base → small → medium → turbo → large-v3) |
| **I** | Toggle Chinese display (ON/OFF) |
| **Q** | Quit gracefully |
| **H** | Show session config + keyboard shortcuts help |

**Note**: Shortcuts respond **instantly** (no need to press Enter).

## Models comparison

| Model | Accuracy | Latency | RAM | Use case |
|-------|----------|---------|-----|----------|
| **tiny** | 60% | ~1.5s | 1GB | Quick tests, low-end systems |
| **base** | 85% | ~4s | 1.5GB | Fast casual listening |
| **small** | 90% | ~5s | 2GB | Good balance |
| **medium** | 95% | ~8s | 5GB | Good CPU balance |
| **large-v3-turbo** | High | Varies | Varies | **Recommended for live translation** |
| **large-v3** | 98% | ~12s | 10GB | Maximum accuracy (high CPU/fan) |

## Speed modes

| Mode | Segment size | Latency | Quality |
|------|--------------|---------|---------|
| **fast** | Shorter | Lower | May cut sentences |
| **normal** | Balanced | Medium | **Default**, good compromise |
| **slow** | Longer | Higher | Complete sentences, best context |

Change mode on-the-fly by pressing **M** during execution.

## Display modes

### English only (default)
```
[14:25:30] ▶ Today was a tough day
[14:25:45] ▶ What happened?
```

### Bilingual (Chinese + English)
```bash
uv run lvt turbo --show-chinese
```
```
[14:25:30] 今天是艰难的一天。                        (green)
[14:25:30] ▶ Today was a difficult day.

[14:25:45] 发生了什么？                            (green)
[14:25:45] ▶ What happened?
```

Toggle Chinese display during execution with **I** key.

## Examples

### Translate YouTube video
```bash
# Start translator
uv run lvt medium

# In another window/tab, open YouTube
firefox "https://www.youtube.com/watch?v=CHINESE_VIDEO_ID"

# Translations appear in real-time in terminal
```

### Translate video call
```bash
# Start with save
uv run lvt medium --save meeting.md

# Join video call (Zoom, Teams, Google Meet, Discord, etc.)
# Translations saved to meeting.md

# During call:
# - Press 'p' to pause (e.g., when speaking)
# - Press 'p' again to resume
# - Press 's' to force save
# - Press 'i' to show Chinese text
```

### Compare models
```bash
# Test tiny (fastest)
uv run lvt tiny --fast

# Test large (best quality)
uv run lvt large --slow
```

## Output format

### Terminal output

**English only:**
```
[14:25:30] ▶ Today was a tough day
[14:25:45] ▶ What happened?
```

**Bilingual (with `--show-chinese` or `i` key):**
```
[14:25:30] 今天是艰难的一天。
[14:25:30] ▶ Today was a difficult day.
[14:25:45] 发生了什么？
[14:25:45] ▶ What happened?
```

### Markdown file (--save)
```markdown
# Live Voice Translation

**Date:** 2026-04-01 14:25:30
**Model:** medium
**Mode:** normal

---

**[14:25:30]**

🇨🇳 *今天是艰难的一天。*

🇬🇧 Today was a tough day

---

**[14:25:45]**

🇨🇳 *发生了什么？*

🇬🇧 What happened?

---

**End of session:** 2026-04-01 14:57:45
**Duration:** 00:32:15
**Phrases:** 147
**Words:** 1823
```

## Audio source selection

The tool finds output devices that currently have playing audio, such as a Chrome tab. If only one device is playing, it is selected automatically. If nothing is playing yet, it listens to the default output device and waits for speech. If several devices are playing, an interactive menu is displayed:

```
  Multiple audio streams detected:

    1) USB Audio
    2) JBL LIVE650BTNC

  Select stream (1-2):
```

PipeWire internal loopback sinks are skipped.

## How it works

1. **Audio source**: Finds the PipeWire output device that is playing audio, with a menu when more than one device is playing
2. **VAD chunking**: webrtcvad detects speech/silence boundaries and flushes each utterance when silence exceeds the mode threshold (500/800/1200ms)
3. **Transcription**: Whisper converts Mandarin audio to Chinese text (`language="zh"`)
4. **Translation**: Argos Translate converts Chinese directly to English (`zh→en`)
5. **Display**: Shows timestamped translations in terminal
6. **Save**: Optionally exports to Markdown file

**Architecture**:
- Audio capture runs in background thread (non-blocking)
- Keyboard controller uses `select()` for instant response (no dependencies)
- Main thread processes audio queue and checks keyboard

## Troubleshooting

### Nothing gets transcribed

```bash
# Show which applications are playing audio
wpctl status

# Restart PipeWire if playback devices are missing
systemctl --user restart pipewire wireplumber
```

---

### Dependency install fails
```bash
# webrtcvad may need Python headers and a compiler if no wheel is available
sudo apt install python3-dev build-essential   # Ubuntu/Debian
sudo dnf install python3-devel gcc             # Fedora/RHEL
sudo zypper install python3-devel gcc          # openSUSE

uv sync
```

### Recreate the environment
```bash
rm -rf .venv
uv sync
```

### Keyboard shortcuts not working

Shortcuts require terminal in TTY mode. If piping output or running in non-interactive environment, use `--no-keyboard`:
```bash
uv run lvt medium --no-keyboard
```

### Chinese text not displaying correctly

Ensure your terminal uses UTF-8 encoding:
```bash
# Check locale
echo $LANG
# Should show: fr_FR.UTF-8, en_US.UTF-8, or similar

# If not UTF-8:
export LANG=fr_FR.UTF-8
export LC_ALL=fr_FR.UTF-8
```

## Performance tips

- **CPU usage**: Use smaller models (tiny/base) on weak hardware
- **Latency**: Use `--fast` (500ms silence, 6s max chunk) for lowest delay — may cut mid-sentence
- **Accuracy**: Use `large --slow` (1200ms silence, 12s max chunk, max beam) for best quality
- **Balance**: `turbo` (default normal mode) is the best live speed/quality tradeoff
- **RAM**: medium model needs ~5GB, large needs ~10GB
- **GPU (NVIDIA)**: Used automatically when CUDA is available. `--cpu` forces CPU transcription. If GPU startup fails, transcription falls back to the CPU.

## Advanced usage

### Custom save filename
```bash
uv run lvt medium --save "$(date +%Y%m%d)-meeting.md"
```

### Create save file during execution
```bash
# Start without --save
uv run lvt medium

# During execution, press 's'
# Creates: live-translate-YYYYMMDD-HHMMSS.md
```

### Wayland compatibility

Works out-of-the-box on Wayland (tested on Fedora 43 + GNOME 49).

### X11 compatibility

Fully compatible with X11 desktop environments.

## License

GNU General Public License v3.0 or later

See [LICENSE](LICENSE) file for details.

## Author

**Stéphane de Labrusse**

Freelance developer specializing in Linux, containerization, and cybersecurity.

- GitHub: [@stephdl](https://github.com/stephdl)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test your changes
4. Submit a pull request

## Roadmap

Potential future features:

- [x] VAD-based real-time chunking (webrtcvad) — flushes on silence, phrase by phrase
- [x] Direct offline Chinese→English translation
- [x] Smart audio source selection with interactive menu for multiple streams
- [x] GPU acceleration (experimental, NVIDIA/CUDA only, auto-fallback to CPU)
- [x] Session statistics - Duration, segments, word count, dropped chunks
- [x] On-the-fly model switching (W key) without restarting
- [x] Markdown export with session stats and bilingual transcript
- [ ] Multiple translators (DeepL/GPT fallback)
- [ ] Bidirectional mode (IT+EN simultaneously)
- [ ] Speaker diarization
- [ ] Web dashboard
- [ ] Export formats (PDF, DOCX, SRT subtitles)
- [ ] Migrate VAD from [webrtcvad](https://github.com/wiseman/py-webrtcvad) (unmaintained since 2018) to a maintained alternative — Silero VAD investigated but requires PyTorch or has ONNX API instability; RMS energy-based VAD is a viable lightweight option

## Acknowledgments

- [Whisper](https://github.com/openai/whisper) by OpenAI
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) by Guillaume Klein
- [Argos Translate](https://github.com/argosopentech/argos-translate) by Argos Open Technologies

## Support

For bugs or feature requests, please [open an issue](https://github.com/stephdl/dev/issues).

---

**Made with ❤️ for the open-source community**
