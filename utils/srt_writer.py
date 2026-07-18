"""
srt_writer.py — SRT subtitle file formatter and writer.

Converts a list of timed transcript segments into a well-formed
.srt file, with intelligent line-length management for readability.
Handles both LTR (English) and RTL (Arabic) text correctly via UTF-8.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, TypedDict


class Segment(TypedDict):
    """A single timed transcript segment from mlx-whisper."""
    start: float   # seconds
    end: float     # seconds
    text: str      # transcript text


# SRT subtitle display guidelines
MAX_CHARS_PER_LINE = 42   # broadcast standard
MAX_LINES_PER_BLOCK = 2   # maximum lines shown at once


def _seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Convert a float number of seconds to SRT timestamp format.

    Example:
        3723.456  →  '01:02:03,456'
    """
    seconds = max(0.0, seconds)
    millis = int(round((seconds % 1) * 1000))
    total_seconds = int(seconds)
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _wrap_text(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> str:
    """
    Wrap subtitle text to fit within the character limit per line.
    Caps output to MAX_LINES_PER_BLOCK lines by truncating if necessary.
    """
    text = text.strip()
    if not text:
        return text

    # Use textwrap to break at word boundaries
    lines = textwrap.wrap(text, width=max_chars, break_long_words=True)

    # Cap to max lines allowed per subtitle block
    if len(lines) > MAX_LINES_PER_BLOCK:
        lines = lines[:MAX_LINES_PER_BLOCK]

    return "\n".join(lines)


def _merge_short_segments(
    segments: List[Segment],
    min_duration: float = 0.5,
) -> List[Segment]:
    """
    Merge very short consecutive segments to avoid subtitle flicker.

    Segments shorter than `min_duration` seconds are merged with
    the next segment if both have similar timing proximity.
    """
    if not segments:
        return segments

    merged: List[Segment] = []
    buffer = dict(segments[0])

    for seg in segments[1:]:
        buf_text = buffer["text"].strip()
        seg_text = seg["text"].strip()

        # Merge if buffer segment is very short and gap to next is tiny
        duration = buffer["end"] - buffer["start"]
        gap = seg["start"] - buffer["end"]

        if duration < min_duration and gap < 0.3:
            buffer["end"] = seg["end"]
            buffer["text"] = buf_text + " " + seg_text
        else:
            if buf_text:
                merged.append(Segment(
                    start=buffer["start"],
                    end=buffer["end"],
                    text=buffer["text"],
                ))
            buffer = dict(seg)

    # Flush the last buffered segment
    if buffer["text"].strip():
        merged.append(Segment(
            start=buffer["start"],
            end=buffer["end"],
            text=buffer["text"],
        ))

    return merged


def segments_to_srt(segments: List[Segment]) -> str:
    """
    Convert a list of timed segments into a complete SRT-formatted string.

    Args:
        segments: List of dicts with keys 'start', 'end', 'text'.

    Returns:
        Full SRT content as a UTF-8 string, ready to write to disk.
    """
    segments = _merge_short_segments(segments)

    srt_blocks: List[str] = []

    for index, seg in enumerate(segments, start=1):
        text = seg["text"].strip()
        if not text:
            continue  # Skip empty segments (silence / noise)

        start_ts = _seconds_to_srt_timestamp(seg["start"])
        end_ts = _seconds_to_srt_timestamp(seg["end"])
        wrapped = _wrap_text(text)

        block = f"{index}\n{start_ts} --> {end_ts}\n{wrapped}"
        srt_blocks.append(block)

    return "\n\n".join(srt_blocks) + "\n"


def write_srt(segments: List[Segment], output_path: Path) -> None:
    """
    Write subtitle segments to an SRT file.

    The file is written with UTF-8 encoding (with BOM marker for maximum
    player compatibility, especially for Arabic text in media players).

    Args:
        segments:    List of timed transcript segments.
        output_path: Destination .srt file path.
    """
    srt_content = segments_to_srt(segments)

    # utf-8-sig adds the BOM (Byte Order Mark) which helps media players
    # like VLC correctly detect Arabic/RTL encoding automatically
    output_path.write_text(srt_content, encoding="utf-8-sig")
