"""
scanner.py — Recursive video file scanner.

Walks a directory tree and returns all video files found,
sorted by their relative path for predictable processing order.
"""

import os
from pathlib import Path
from typing import List

# All video container formats we handle
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".m4v",
    ".webm",
    ".flv",
    ".ts",
    ".wmv",
    ".mts",
    ".m2ts",
    ".3gp",
    ".ogv",
}


def scan_videos(root: str | Path) -> List[Path]:
    """
    Recursively scan `root` for video files.

    Args:
        root: Path to the top-level folder to scan.

    Returns:
        Sorted list of absolute Path objects for each video found.

    Raises:
        NotADirectoryError: If `root` does not exist or is not a directory.
    """
    root = Path(root).expanduser().resolve()

    if not root.exists():
        raise NotADirectoryError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    videos: List[Path] = []

    for dirpath, _dirnames, filenames in os.walk(root):
        # Sort filenames for deterministic order within each directory
        for filename in sorted(filenames):
            filepath = Path(dirpath) / filename
            if filepath.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(filepath.resolve())

    # Sort the full list by relative path so nested folders are grouped
    videos.sort(key=lambda p: p.relative_to(root))
    return videos


def subtitle_path(video: Path) -> Path:
    """
    Return the expected .srt path for a given video file.

    Example:
        /videos/lecture01.mp4  →  /videos/lecture01.srt
    """
    return video.with_suffix(".srt")


def needs_transcription(video: Path) -> bool:
    """Return True if the video has no existing .srt subtitle file."""
    return not subtitle_path(video).exists()
