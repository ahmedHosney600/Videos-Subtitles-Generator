"""
transcriber.py — MLX-Whisper transcription engine (Apple Silicon optimized).

Loads the model ONCE at startup and reuses it across all videos.
Audio is extracted from each video via ffmpeg into a temporary 16kHz
mono WAV file before being passed to the model for transcription.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from utils.srt_writer import Segment


import sys

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# MLX-compatible model IDs hosted on Hugging Face (mlx-community namespace).
# These are quantized Apple MLX builds that run natively on M-series GPUs.
MODELS_MLX = {
    "large-v3":       "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "arabic-v3":      "__local__",   # resolved at runtime
}

# Faster-Whisper model IDs (either HF repos or local paths).
MODELS_FASTER_WHISPER = {
    "large-v3":       "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "arabic-v3":      "__local__",   # resolved at runtime
}

# Absolute path to the locally-converted Arabic fine-tuned model
_ARABIC_LOCAL_MODEL_MLX = (
    Path(__file__).parent.parent / "models" / "whisper-large-v3-arabic-mlx"
)
_ARABIC_LOCAL_MODEL_CT2 = (
    Path(__file__).parent.parent / "models" / "whisper-large-v3-arabic-ct2"
)


def get_model_path(model_key: str) -> str:
    """
    Resolve a model key to a HuggingFace repo ID or a local path string.
    """
    if sys.platform == "darwin":
        if model_key == "arabic-v3":
            if not (_ARABIC_LOCAL_MODEL_MLX / "weights.safetensors").exists():
                raise RuntimeError(
                    "Arabic fine-tuned model not found.\n"
                    "Run the converter first:\n"
                    "  source .venv/bin/activate\n"
                    "  python3 convert_arabic_model.py"
                )
            return str(_ARABIC_LOCAL_MODEL_MLX)
        return MODELS_MLX[model_key]
    else:
        if model_key == "arabic-v3":
            if not (_ARABIC_LOCAL_MODEL_CT2 / "model.bin").exists():
                raise RuntimeError(
                    "Arabic fine-tuned model not found in CTranslate2 format.\n"
                    "Run the converter first:\n"
                    "  python3 convert_arabic_model.py"
                )
            return str(_ARABIC_LOCAL_MODEL_CT2)
        return MODELS_FASTER_WHISPER[model_key]


def is_arabic_model_ready() -> bool:
    """Return True if the locally-converted Arabic model is available."""
    if sys.platform == "darwin":
        return (_ARABIC_LOCAL_MODEL_MLX / "weights.safetensors").exists()
    else:
        return (_ARABIC_LOCAL_MODEL_CT2 / "model.bin").exists()


# Language codes used by Whisper (ISO 639-1)
LANGUAGE_CODES = {
    "english": "en",
    "arabic": "ar",
}


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, output_wav: str) -> None:
    """
    Extract the audio track from a video file and write it as a
    16 kHz mono WAV file — the exact format Whisper expects.

    Uses ffmpeg subprocess directly for reliability and broad codec support.

    Args:
        video_path:  Path to the source video file.
        output_wav:  Destination path for the extracted WAV file.

    Raises:
        RuntimeError: If ffmpeg is not installed or extraction fails.
    """
    cmd = [
        "ffmpeg",
        "-y",                    # Overwrite output without asking
        "-i", str(video_path),   # Input video
        "-vn",                   # Strip video stream (audio only)
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian (WAV)
        "-ar", "16000",          # 16 kHz sample rate (Whisper requirement)
        "-ac", "1",              # Mono channel
        "-loglevel", "error",    # Suppress ffmpeg chatter
        output_wav,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it with: brew install ffmpeg"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from '{video_path.name}':\n{e.stderr}"
        )


# ---------------------------------------------------------------------------
# Transcription engine
# ---------------------------------------------------------------------------

class Transcriber:
    """
    Wraps mlx-whisper to provide a simple per-video transcription interface.

    The model is loaded once at construction time and reused for all
    subsequent calls to `transcribe()`, avoiding expensive reload overhead.

    Example:
        engine = Transcriber(model_key="large-v3", language="arabic")
        segments = engine.transcribe(Path("/videos/lecture.mp4"))
    """

    def __init__(self, model_key: str = "large-v3", language: str = "english") -> None:
        """
        Initialize and load the model based on platform.

        Args:
            model_key: One of 'large-v3' or 'large-v3-turbo'.
            language:  'english' or 'arabic'.
        """
        valid_keys = MODELS_MLX if sys.platform == "darwin" else MODELS_FASTER_WHISPER
        if model_key not in valid_keys:
            raise ValueError(
                f"Unknown model '{model_key}'. Choose from: {list(valid_keys.keys())}"
            )
        if language.lower() not in LANGUAGE_CODES:
            raise ValueError(
                f"Unsupported language '{language}'. Choose from: {list(LANGUAGE_CODES.keys())}"
            )

        self.model_id = get_model_path(model_key)
        self.language_code = LANGUAGE_CODES[language.lower()]
        self._engine = None  # Lazy import after user sees the loading message

    def load(self) -> None:
        """
        Import the respective engine (mlx-whisper or faster-whisper) and pre-warm the model.
        Called explicitly so the caller can show a loading indicator first.
        """
        if sys.platform == "darwin":
            try:
                import mlx_whisper  # type: ignore[import]
                self._engine = mlx_whisper
            except ImportError:
                raise RuntimeError(
                    "mlx-whisper is not installed.\n"
                    "Install it with: pip install mlx-whisper"
                )
        else:
            try:
                from faster_whisper import WhisperModel
                # Colab instances typically have Nvidia GPUs, we use FP16 for speed
                self._engine = WhisperModel(self.model_id, device="cuda", compute_type="float16")
            except ImportError:
                raise RuntimeError(
                    "faster-whisper is not installed.\n"
                    "Install it with: pip install faster-whisper ctranslate2"
                )

    def transcribe(
        self,
        video_path: Path,
        progress_callback: Optional[callable] = None,
    ) -> List[Segment]:
        """
        Transcribe a single video file and return timed subtitle segments.

        This method:
          1. Extracts audio from the video via ffmpeg
          2. Runs the MLX-Whisper model on the extracted audio
          3. Returns a cleaned list of Segment dicts

        Args:
            video_path:         Path to the video file.
            progress_callback:  Optional callable(float) called with progress 0–1.

        Returns:
            List of Segment dicts: [{'start': float, 'end': float, 'text': str}]

        Raises:
            RuntimeError: If model is not loaded or audio extraction fails.
        """
        if self._engine is None:
            raise RuntimeError("Model not loaded. Call Transcriber.load() first.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Step 1: Extract audio
            extract_audio(video_path, tmp_path)

            # Step 2: Transcribe with selected engine
            raw_segments = []

            if sys.platform == "darwin":
                result = self._engine.transcribe(
                    tmp_path,
                    path_or_hf_repo=self.model_id,
                    language=self.language_code,
                    word_timestamps=True,
                    condition_on_previous_text=True,
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    verbose=False,
                )
                raw_segments = result.get("segments", [])
            else:
                segments_iter, _ = self._engine.transcribe(
                    tmp_path,
                    language=self.language_code,
                    word_timestamps=True,
                    condition_on_previous_text=True,
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                )
                for seg in segments_iter:
                    raw_segments.append({
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                    })

        finally:
            # Always clean up the temp audio file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Step 3: Convert raw segment dictionaries to our Segment format
        segments: List[Segment] = []

        for seg in raw_segments:
            text = seg.get("text", "").strip()
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))

            # Skip empty, no-speech, or zero-duration segments
            if not text or end <= start:
                continue

            # Skip common Whisper hallucination artifacts
            if _is_hallucination(text):
                continue

            segments.append(Segment(start=start, end=end, text=text))

        return segments


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

# Common Whisper hallucination phrases that appear on silent audio
_HALLUCINATION_PHRASES = {
    "thank you",
    "thanks for watching",
    "thanks for watching!",
    "please subscribe",
    "like and subscribe",
    "[music]",
    "[applause]",
    "[laughter]",
    "[ music ]",
    "[ applause ]",
    "(music)",
    "(applause)",
    "subtitles by",
    "transcribed by",
    "www.",
    "http",
}


def _is_hallucination(text: str) -> bool:
    """
    Detect common Whisper hallucination patterns.

    Returns True if the text looks like a hallucinated artifact
    that should be dropped from the subtitle output.
    """
    lowered = text.lower().strip().rstrip(".")

    # Exact match against known hallucination phrases
    if lowered in _HALLUCINATION_PHRASES:
        return True

    # Catch phrases that contain hallucination markers
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in lowered and len(lowered) < 60:
            return True

    # Repetition detection: if the same word appears > 5 times consecutively
    words = lowered.split()
    if len(words) >= 6:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:  # Less than 25% unique words = likely looping
            return True

    return False
