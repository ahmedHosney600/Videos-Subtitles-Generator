# 🎬 Video → Subtitles

> Recursively transcribe video folders into SRT subtitle files using  
> **MLX-Whisper Large-v3** — Apple Silicon native, runs on your M1 GPU.

Supports **English** and **Arabic**. Processes entire folder trees, skips already-subtitled videos, and saves each `.srt` file right next to its video.

---

## ✨ Features

- 🍎 **Apple Silicon optimized** — uses MLX (Apple's ML framework), not CPU-only libraries
- 🤖 **Whisper Large-v3** — highest accuracy model for both English and Arabic
- 🌐 **Arabic-ready** — UTF-8-BOM encoding ensures correct display in all media players
- 📁 **Recursive scanning** — handles any depth of nested folders
- ⏭️  **Smart skip** — skips videos that already have an `.srt` file
- 🛡️  **Hallucination filter** — removes common Whisper artifacts from silent audio
- 📊 **Rich terminal UI** — real-time progress bars and final summary
- 📋 **Optional log file** — saves a `transcription_log.txt` with full results

---

## 🖥️ Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+**
- **ffmpeg** (for audio extraction)

---

## ⚙️ Installation

### 1. Install ffmpeg

```bash
brew install ffmpeg
```

### 2. Create a virtual environment (recommended)

```bash
cd "Video To Subtitles"
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `mlx-whisper` will automatically download the model weights (~3 GB for Large-v3)  
> on the **first run**. Subsequent runs load from cache instantly.

---

## 🚀 Usage

### Interactive mode (recommended)

```bash
python transcribe.py
```

You'll be guided through three prompts:

```
📁 Enter path to folder containing videos: /path/to/my/videos

🌐 Select language:
   [1] 🇬🇧  English
   [2] 🇸🇦  Arabic
➤ Choice: 2

⚡ Select model:
   [1] Whisper Large-v3         ⚡⚡    ★★★★★
   [2] Whisper Large-v3 Turbo   ⚡⚡⚡⚡  ★★★★½
➤ Choice: 1
```

### CLI flags (for scripting / automation)

```bash
# Pass all options as arguments (no prompts)
python transcribe.py \
  --folder /path/to/videos \
  --language arabic \
  --model large-v3

# Re-transcribe videos that already have .srt files
python transcribe.py --force

# Save a transcription_log.txt in the target folder
python transcribe.py --log

# Combine flags
python transcribe.py --folder /videos --language english --model large-v3-turbo --force --log
```

---

## 📂 Output

For each video, an `.srt` file is created in the **same folder** as the video:

```
📁 /videos/
├── 📁 lectures/
│   ├── 🎬 lecture_01.mp4
│   ├── 📄 lecture_01.srt   ← created by this tool
│   ├── 🎬 lecture_02.mkv
│   └── 📄 lecture_02.srt   ← created by this tool
├── 🎬 intro.mp4
└── 📄 intro.srt            ← created by this tool
```

### SRT format example

```srt
1
00:00:01,240 --> 00:00:04,820
Welcome to this lecture on machine learning.

2
00:00:05,100 --> 00:00:08,630
Today we'll cover neural network architectures.
```

---

## 🤖 Model Details

| Model | HF Repo | Speed | Accuracy | VRAM |
|-------|---------|-------|----------|------|
| `large-v3` | `mlx-community/whisper-large-v3-mlx` | ⚡⚡ | ★★★★★ | ~3 GB |
| `large-v3-turbo` | `mlx-community/whisper-large-v3-turbo` | ⚡⚡⚡⚡ | ★★★★½ | ~1.5 GB |

**Why MLX?** Unlike `faster-whisper` (which uses CTranslate2 — CPU only on Mac),  
`mlx-whisper` natively uses your M1's unified GPU and Neural Engine for  
significantly faster transcription.

---

## 🌐 Supported Video Formats

`.mp4` · `.mkv` · `.mov` · `.avi` · `.m4v` · `.webm` · `.flv`  
`.ts` · `.wmv` · `.mts` · `.m2ts` · `.3gp` · `.ogv`

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ffmpeg not found` | Run `brew install ffmpeg` |
| `mlx-whisper not found` | Run `pip install mlx-whisper` |
| Model download hangs | Check internet connection; ~3 GB download on first run |
| Arabic text looks wrong in player | Ensure your media player supports UTF-8-BOM SRT files |
| Poor accuracy on a dialect | Try `large-v3` if using turbo, or use `--force` after switching models |
| `No video files found` | Check that the folder path is correct and contains supported formats |
# Videos-Subtitles-Generator
