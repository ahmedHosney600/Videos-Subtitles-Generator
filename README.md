# 🎬 Video → Subtitles

> Recursively transcribe video folders into SRT subtitle files using  
> **Whisper Large-v3** — runs on **Apple Silicon** (MLX) and **Google Colab** (CUDA).

Supports **English** and **Arabic** (including Egyptian dialect). Processes entire folder trees, skips already-subtitled videos, and saves each `.srt` file right next to its video.

---

## ✨ Features

- 🍎 **Apple Silicon optimized** — uses MLX (Apple's ML framework) for native GPU inference
- ☁️ **Google Colab support** — works on free T4 GPUs via `faster-whisper`
- 🇸🇦 **Arabic Fine-tuned model** — optional `Byne/whisper-large-v3-arabic` for best dialect accuracy
- 📁 **Recursive scanning** — handles any depth of nested folders
- ⏭️ **Smart skip** — skips videos that already have an `.srt` file
- 🛡️ **Hallucination filter** — removes common Whisper artifacts from silent audio
- 📊 **Rich terminal UI** — real-time progress bars and final summary
- 📋 **Optional log file** — saves a `transcription_log.txt` with full results

---

## 🖥️ Requirements

### Local (macOS)
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- ffmpeg

### Google Colab
- Free Google account (T4 GPU runtime)
- Videos uploaded to Google Drive

---

## ⚙️ Installation (Local)

### 1. Install ffmpeg

```bash
brew install ffmpeg
```

### 2. Create a virtual environment

```bash
cd "Video To Subtitles"
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Models are downloaded automatically on first run (~3 GB for Large-v3).

---

## ☁️ Google Colab

Open `Colab_Transcription.ipynb` in Google Colab and follow the 4 steps:

1. Mount Google Drive
2. Setup environment (installs ffmpeg + dependencies)
3. *(Optional)* Convert Arabic Fine-tuned model
4. Run transcription

The notebook uses `faster-whisper` with CUDA on the free T4 GPU.

---

## 🚀 Usage

### Interactive mode (recommended)

```bash
python transcribe.py
```

You'll be guided through four steps:

```
Step 1 — 📁 Folder:     /path/to/my/videos
Step 2 — 🌐 Language:   Arabic
Step 3 — 🤖 Model:      Large V3 Turbo
Step 4 — 🔧 Optimisations:
           GPU memory:  75%
           Caffeinate:  yes
```

### CLI flags (for scripting / automation)

```bash
# Full CLI (no interactive prompts)
python transcribe.py \
  --folder /path/to/videos \
  --language arabic \
  --model arabic-v3 \
  --gpu-mem 85 \
  --caffeinate

# Re-transcribe videos that already have .srt files
python transcribe.py --force

# Save a transcription log
python transcribe.py --log

# Disable caffeinate
python transcribe.py --model large-v3 --no-caffeinate
```

| Flag | Description |
|------|-------------|
| `--folder PATH` | Target folder (skips prompt) |
| `--language` | `english` or `arabic` |
| `--model` | `large-v3`, `large-v3-turbo`, or `arabic-v3` |
| `--gpu-mem N` | GPU memory limit % (25–100, default: 75) |
| `--caffeinate` | Prevent macOS thermal throttling (default: yes) |
| `--no-caffeinate` | Disable caffeinate |
| `--force` | Re-transcribe even if `.srt` exists |
| `--log` | Save `transcription_log.txt` |

---

## 📂 Output

For each video, an `.srt` file is created in the **same folder** as the video:

```
📁 /videos/
├── 📁 lectures/
│   ├── 🎬 lecture_01.mp4
│   ├── 📄 lecture_01.srt   ← generated
│   ├── 🎬 lecture_02.mkv
│   └── 📄 lecture_02.srt   ← generated
├── 🎬 intro.mp4
└── 📄 intro.srt            ← generated
```

---

## 🤖 Models

| Model | Description | Speed | Accuracy |
|-------|-------------|-------|----------|
| `large-v3` | Best accuracy for clear speech | ⚡⚡ | ★★★★★ |
| `large-v3-turbo` | ~2× faster, slightly lower accuracy | ⚡⚡⚡⚡ | ★★★★½ |
| `arabic-v3` | Fine-tuned for Arabic dialects (Egyptian, Gulf, etc.) | ⚡⚡ | ★★★★★ |

### Arabic Fine-tuned Model

The `arabic-v3` model uses [Byne/whisper-large-v3-arabic](https://huggingface.co/Byne/whisper-large-v3-arabic), fine-tuned on Arabic speech data. To use it:

```bash
python3 convert_arabic_model.py   # One-time conversion (~2 min)
python3 transcribe.py             # Select "Arabic Fine-tuned" in Step 3
```

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
| Poor accuracy on a dialect | Try `arabic-v3` model or `large-v3` instead of turbo |
| `No video files found` | Check that the folder path is correct and contains supported formats |
| Speed drops during transcription | Normal — Whisper retries difficult audio segments at higher temperatures |
