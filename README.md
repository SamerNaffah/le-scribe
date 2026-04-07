# Le Scribe 📜

A fully local Google Meet transcription bot for macOS (Apple Silicon).

- Joins Google Meet as a guest bot — no Google account required
- Captures audio from both remote participants (via BlackHole) and your own mic
- Transcribes in real-time with **mlx-whisper** — fully on-device, no cloud API
- **Speaker diarization** — knows who said what (SPEAKER_00, SPEAKER_01, ...)
- Live transcript visible in a web UI at `http://localhost:5050`
- Saves full session transcript as `.txt`, `.json`, and `.pdf`

---

## Requirements

| Requirement | Notes |
|---|---|
| macOS (Apple Silicon) | M1/M2/M3/M4 — mlx-whisper is Apple Silicon only |
| Python 3.11+ | `brew install python@3.11` |
| Homebrew | https://brew.sh |
| Google Chrome | Must be installed |
| BlackHole 2ch | Virtual audio loopback — `brew install blackhole-2ch` |
| HuggingFace account | Free — required for speaker diarization model |

---

## Installation

### 1. Install Python dependencies

```bash
cd meet-transcriber
bash setup.sh
```

### 2. Set up speaker diarization (HuggingFace)

Speaker diarization uses pyannote.audio, which requires accepting model terms:

1. Create a free account at https://huggingface.co
2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accept terms at https://huggingface.co/pyannote/segmentation-3.0
4. Generate a token at https://huggingface.co/settings/tokens (read access)
5. Add your token to `transcriber.py`:
   ```python
   _HF_TOKEN = "your_token_here"
   ```

### 2. Install BlackHole

```bash
brew install blackhole-2ch
```

Reboot after installing so macOS registers the new audio device.

### 3. Fix Python SSL certificates (Python 3.14+ only)

```bash
open "/Applications/Python 3.14/Install Certificates.command"
```

### 4. Install ChromeDriver

The bot needs a ChromeDriver binary that matches your Chrome version.

1. Check your Chrome version at `chrome://settings/help`
2. Download the matching arm64 ChromeDriver from https://googlechromelabs.github.io/chrome-for-testing/
3. Place it at:
   ```
   ~/Library/Application Support/meet_transcriber/chromedriver
   ```
4. Make it executable and sign it:
   ```bash
   chmod +x ~/Library/Application\ Support/meet_transcriber/chromedriver
   xattr -cr ~/Library/Application\ Support/meet_transcriber/chromedriver
   codesign --sign - --force ~/Library/Application\ Support/meet_transcriber/chromedriver
   ```
5. Open it once so macOS prompts you to allow it:
   ```bash
   open ~/Library/Application\ Support/meet_transcriber/chromedriver
   ```
   Then go to **System Settings → Privacy & Security** and click **Allow Anyway**.

### 5. Configure Audio MIDI Setup (one-time)

This routes meeting audio through BlackHole so the bot can capture it.

1. Open **Spotlight** → **Audio MIDI Setup**
2. Click **+** → **Create Multi-Output Device**
3. Check both:
   - **BlackHole 2ch** ← set this as the master clock device
   - Your speakers or headphones
4. Set the sample rate to **48000 Hz**
5. Open **System Settings → Sound → Output** and select the Multi-Output Device

> **Important:** The master clock device in the Multi-Output Device must be **BlackHole 2ch** (not your headphones), otherwise the sample rate will be forced to your headphone's rate and audio capture will fail.

---

## Running

### Web UI (recommended)

```bash
python server.py
```

Opens a browser at `http://localhost:5050`. Paste your Meet URL, choose language, and click Start.

### Command line

```bash
python bot.py --meet https://meet.google.com/xxx-xxxx-xxx --lang auto
```

---

## How it works

1. Chrome opens as a guest and navigates to the Meet URL
2. The bot enters the name **"Transcription Bot"** and asks to join
3. The meeting host admits the bot
4. Audio capture starts — two streams are mixed:
   - **BlackHole**: captures audio from other participants (played by the bot's Chrome)
   - **Mic**: captures your own voice (never echoed back by Meet)
5. VAD (Silero) detects when someone finishes speaking and triggers transcription
6. Pyannote diarizes the chunk (who spoke), Whisper transcribes it (what was said):
   ```
   [00:00:08] (fr) [SPEAKER_00] Bonjour tout le monde, bienvenue à cette réunion.
   [00:00:15] (en) [SPEAKER_01] Let's start with the agenda for today.
   ```
6. Stop the session — transcript is saved automatically

---

## CLI options

```
--meet URL               Google Meet URL (required)
--lang CODE              Language code ('fr', 'en', etc.) or 'auto' (default: auto)
--bot-name NAME          Display name in the meeting (default: Transcription Bot)
--chunk-seconds N        Audio chunk size in seconds (default: 10)
--silence-threshold F    RMS cutoff below which chunks are skipped (default: 0.005)
--admission-timeout N    Seconds to wait to be admitted (default: 300)
```

---

## Transcript output

Saved to `~/Documents/transcriptions/`:

**Text** (`YYYY-MM-DD_HH-MM_meet.txt`):
```
Google Meet Transcript — 2026-04-07 17:53
============================================================

[00:00:10] (fr) Bonjour tout le monde, bienvenue à cette réunion.
[00:00:20] (en) Let's start with the agenda for today.
```

**JSON** (`YYYY-MM-DD_HH-MM_meet.json`):
```json
[
  {"time": "00:00:10", "lang": "fr", "text": "Bonjour tout le monde..."},
  {"time": "00:00:20", "lang": "en", "text": "Let's start with the agenda..."}
]
```

PDF export is also available from the web UI.

---

## Troubleshooting

### All audio chunks skipped as silence (RMS=0.00000)
- Make sure the Multi-Output Device is set as system output in Sound settings
- Make sure **BlackHole 2ch** is the master clock in the Multi-Output Device (sets sample rate to 48kHz)
- Play audio and verify you hear it through your speakers/headphones

### Chrome opens and immediately closes
- ChromeDriver version must match Chrome version exactly
- Re-run the `xattr` + `codesign` steps from Installation
- Open chromedriver once manually and allow it in Privacy & Security

### Stuck at "Waiting to be admitted"
- The host needs to admit the bot from the meeting participants panel
- Increase timeout: `--admission-timeout 600`

### Model download on first run
`mlx-community/whisper-large-v3-turbo` (~800 MB) downloads once to `~/.cache/huggingface/`. Pre-download it before your first meeting:
```bash
python3 -c "import mlx_whisper, numpy as np; mlx_whisper.transcribe(np.zeros(16000, dtype='float32'), path_or_hf_repo='mlx-community/whisper-large-v3-turbo')"
```

---

## Contributing

This project is actively evolving. We're looking to add new features and improve the overall experience — contributions, ideas, and feedback are very welcome.

Some directions we're exploring:
- Speaker diarization (who said what)
- Automatic meeting summary generation
- Support for other meeting platforms (Zoom, Teams)
- Zoom and calendar integrations
- Improved noise filtering and audio quality

Feel free to open an issue, submit a pull request, or reach out directly if you want to collaborate.
