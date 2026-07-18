#!/usr/bin/env python3
"""
transcribe.py — Video → Subtitles CLI
======================================
Recursively scans a folder for video files and transcribes each one
into an SRT subtitle file using MLX-Whisper (Apple Silicon optimized).

Usage:
    python transcribe.py
    python transcribe.py --force      # Re-transcribe even if .srt exists
    python transcribe.py --log        # Save transcription_log.txt to folder

Supported languages: English, Arabic
Supported video formats: mp4, mkv, mov, avi, m4v, webm, flv, ts, wmv, mts, m2ts, 3gp
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from utils.scanner import needs_transcription, scan_videos, subtitle_path
from utils.srt_writer import write_srt
from utils.transcriber import Transcriber, is_arabic_model_ready

# ---------------------------------------------------------------------------
# Speed mode definitions
# Each mode bundles: model, GPU memory limit, MPS fallback, caffeinate flag
# ---------------------------------------------------------------------------

SPEED_MODES: Dict[str, dict] = {
    "quality": {
        "label":           "Quality",
        "model":           "large-v3",
        "gpu_mem_limit":   "0.75",
        "mps_fallback":    False,
        "caffeinate":      False,
        "description":     "Highest accuracy · Large-v3 · 75% GPU memory",
        "speed_icon":      "⚡⚡",
        "quality_icon":    "★★★★★",
        "arabic_only":     False,
    },
    "balanced": {
        "label":           "Balanced",
        "model":           "large-v3-turbo",
        "gpu_mem_limit":   "0.75",
        "mps_fallback":    True,
        "caffeinate":      False,
        "description":     "~2× faster · Turbo model · 75% GPU memory · MPS fallback",
        "speed_icon":      "⚡⚡⚡",
        "quality_icon":    "★★★★½",
        "arabic_only":     False,
    },
    "fast": {
        "label":           "Fast",
        "model":           "large-v3-turbo",
        "gpu_mem_limit":   "0.75",
        "mps_fallback":    True,
        "caffeinate":      True,
        "description":     "Maximum speed · Turbo · 75% GPU memory · MPS · no throttle",
        "speed_icon":      "⚡⚡⚡⚡",
        "quality_icon":    "★★★★½",
        "arabic_only":     False,
    },
    "arabic-fine-tuned": {
        "label":           "Arabic Fine-tuned",
        "model":           "arabic-v3",
        "gpu_mem_limit":   "0.75",
        "mps_fallback":    True,
        "caffeinate":      False,
        "description":     "Fine-tuned on Arabic · Best for dialects · Byne/whisper-large-v3-arabic",
        "speed_icon":      "⚡⚡",
        "quality_icon":    "★★★★★",
        "arabic_only":     True,    # Only makes sense for Arabic language
    },
}

# ---------------------------------------------------------------------------
# Console / Theme
# ---------------------------------------------------------------------------

THEME = Theme(
    {
        "primary":   "bold cyan",
        "secondary": "dim cyan",
        "success":   "bold green",
        "warning":   "bold yellow",
        "error":     "bold red",
        "info":      "dim white",
        "accent":    "bold magenta",
        "muted":     "dim",
        "file":      "italic yellow",
    }
)

console = Console(theme=THEME, highlight=False)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = """
[bold cyan]██╗   ██╗██╗██████╗ ███████╗ ██████╗     ██████╗ ██╗   ██╗██████╗ ███████╗[/]
[bold cyan]██║   ██║██║██╔══██╗██╔════╝██╔═══██╗    ██╔══██╗██║   ██║██╔══██╗██╔════╝[/]
[bold cyan]██║   ██║██║██║  ██║█████╗  ██║   ██║    ██║  ██║██║   ██║██████╔╝███████╗[/]
[cyan]╚██╗ ██╔╝██║██║  ██║██╔══╝  ██║   ██║    ██║  ██║██║   ██║██╔══██╗╚════██║[/]
[cyan] ╚████╔╝ ██║██████╔╝███████╗╚██████╔╝    ██████╔╝╚██████╔╝██████╔╝███████║[/]
[dim cyan]  ╚═══╝  ╚═╝╚═════╝ ╚══════╝ ╚═════╝     ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝[/]
"""

SUBTITLE_LINE = "[dim cyan]  ✦  Apple Silicon Optimized  ✦  MLX Whisper Large-v3  ✦  Arabic & English  ✦[/]"


def print_banner() -> None:
    console.print()
    console.print(BANNER)
    console.print(SUBTITLE_LINE, justify="center")
    console.print()


# ---------------------------------------------------------------------------
# Interactive setup prompts
# ---------------------------------------------------------------------------

def prompt_folder() -> Path:
    """Ask user for the target folder path and validate it."""
    console.print(Rule("[secondary]Step 1 — Target Folder[/]"))
    console.print()

    while True:
        raw = Prompt.ask(
            "[primary]📁 Enter path to folder containing videos[/]",
            console=console,
        ).strip()

        if not raw:
            console.print("  [error]Path cannot be empty. Please try again.[/]")
            continue

        path = Path(raw).expanduser().resolve()

        if not path.exists():
            console.print(f"  [error]Path does not exist:[/] [file]{path}[/]")
            continue
        if not path.is_dir():
            console.print(f"  [error]Path is not a directory:[/] [file]{path}[/]")
            continue

        console.print(f"  [success]✓[/] Using folder: [file]{path}[/]")
        console.print()
        return path


def prompt_language() -> str:
    """Ask user to pick the video language."""
    console.print(Rule("[secondary]Step 2 — Video Language[/]"))
    console.print()

    table = Table(box=box.ROUNDED, show_header=False, border_style="dim cyan", padding=(0, 2))
    table.add_column("Key", style="bold cyan", width=5)
    table.add_column("Language", style="white")
    table.add_column("Note", style="dim")
    table.add_row("1", "🇬🇧  English", "Optimized for clear English speech")
    table.add_row("2", "🇸🇦  Arabic", "MSA + major dialects supported")
    console.print(table)
    console.print()

    while True:
        choice = Prompt.ask(
            "[primary]🌐 Select language[/]",
            choices=["1", "2"],
            console=console,
        )
        lang_map = {"1": "english", "2": "arabic"}
        language = lang_map[choice]
        console.print(f"  [success]✓[/] Language set to: [accent]{language.title()}[/]")
        console.print()
        return language


def prompt_speed_mode() -> dict:
    """Ask user to choose a speed mode that bundles model + all optimisations."""
    console.print(Rule("[secondary]Step 3 — Speed Mode[/]"))
    console.print()

    arabic_ready = is_arabic_model_ready()

    table = Table(box=box.ROUNDED, border_style="dim cyan", padding=(0, 2))
    table.add_column("#",        style="bold cyan",  width=3,  justify="center")
    table.add_column("Mode",     style="bold white", width=20)
    table.add_column("Speed",    style="yellow",     width=8,  justify="center")
    table.add_column("Accuracy", style="green",      width=10, justify="center")
    table.add_column("Model",    style="cyan",       width=22)
    table.add_column("Optimisations applied", style="dim")

    # Build list of available modes (arabic-fine-tuned only shown if converted)
    mode_keys = [
        k for k in SPEED_MODES
        if k != "arabic-fine-tuned" or arabic_ready
    ]

    for i, key in enumerate(mode_keys, start=1):
        m = SPEED_MODES[key]
        opts = []
        opts.append(f"GPU mem {int(float(m['gpu_mem_limit'])*100)}%")
        if m["mps_fallback"]:
            opts.append("MPS fallback")
        if m["caffeinate"]:
            opts.append("caffeinate (no throttle)")

        label = m["label"]
        if key == "arabic-fine-tuned":
            label = "🇸🇦 " + label   # Saudi flag for visual distinction

        table.add_row(
            str(i),
            label,
            m["speed_icon"],
            m["quality_icon"],
            m["model"],
            " · ".join(opts),
        )

    console.print(table)

    # Show a hint if the Arabic model isn't converted yet
    if not arabic_ready:
        console.print(
            "  [dim]💡 Tip: run [bold]python3 convert_arabic_model.py[/] to unlock"
            " the [bold]Arabic Fine-tuned[/] mode (best for dialects)[/]"
        )

    console.print()

    choice = Prompt.ask(
        "[primary]⚡ Select speed mode[/]",
        choices=[str(i) for i in range(1, len(mode_keys) + 1)],
        default="1",
        console=console,
    )
    selected_key = mode_keys[int(choice) - 1]
    mode = SPEED_MODES[selected_key]

    console.print(
        f"  [success]✓[/] Mode: [accent]{mode['label']}[/]  "
        f"[dim]({mode['description']})[/]"
    )
    console.print()
    return mode


def confirm_start(
    folder: Path,
    language: str,
    speed_mode: dict,
    total_videos: int,
    to_process: int,
    skipped_existing: int,
) -> bool:
    """Show a configuration summary and ask user to confirm."""
    console.print(Rule("[secondary]Summary — Ready to Start[/]"))
    console.print()

    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="dim", min_width=22)
    grid.add_column(style="white")

    opts = [f"GPU {int(float(speed_mode['gpu_mem_limit'])*100)}% memory"]
    if speed_mode["mps_fallback"]:
        opts.append("MPS fallback")
    if speed_mode["caffeinate"]:
        opts.append("caffeinate")

    grid.add_row("📁 Folder",        str(folder))
    grid.add_row("🌐 Language",      language.title())
    grid.add_row("⚡ Speed Mode",    f"{speed_mode['label']}  {speed_mode['speed_icon']}")
    grid.add_row("🤖 Model",         speed_mode["model"])
    grid.add_row("🔧 Optimisations", " · ".join(opts))
    grid.add_row("🎬 Videos found",  str(total_videos))
    grid.add_row("🔄 To transcribe", f"[bold]{to_process}[/]")
    if skipped_existing > 0:
        grid.add_row(
            "⏭  Already done",
            f"[dim]{skipped_existing} (already have .srt)[/]",
        )

    console.print(
        Panel(grid, border_style="cyan", title="[bold cyan]Configuration[/]", padding=(1, 3))
    )
    console.print()

    if to_process == 0:
        console.print(
            "[warning]⚠  All videos already have subtitle files. "
            "Use --force to re-transcribe.[/]"
        )
        return False

    answer = Prompt.ask(
        "[primary]▶  Start transcription?[/]",
        choices=["y", "n"],
        default="y",
        console=console,
    )
    console.print()
    return answer.lower() == "y"


# ---------------------------------------------------------------------------
# Progress display helpers
# ---------------------------------------------------------------------------

def make_overall_progress() -> Progress:
    """Create the overall (per-video) progress bar."""
    return Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None, style="cyan", complete_style="bold cyan"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )


def make_video_progress() -> Progress:
    """Create the per-video transcription progress bar."""
    return Progress(
        TextColumn("  [dim]↳[/]  [italic dim]{task.description}[/]"),
        SpinnerColumn(spinner_name="arc", style="dim cyan"),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    )


# ---------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------

class TranscriptionLog:
    """Accumulates per-video results and writes a summary log file."""

    def __init__(self, folder: Path, language: str, model_key: str) -> None:
        self.folder = folder
        self.language = language
        self.model_key = model_key
        self.start_time = datetime.now()
        self.entries: List[dict] = []

    def record(
        self,
        video: Path,
        status: str,
        duration_s: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        self.entries.append(
            {
                "video": str(video.relative_to(self.folder)),
                "status": status,
                "duration_s": duration_s,
                "error": error,
            }
        )

    def write(self, output_path: Path) -> None:
        lines = [
            "=" * 70,
            "  VIDEO → SUBTITLES — TRANSCRIPTION LOG",
            "=" * 70,
            f"  Date       : {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Folder     : {self.folder}",
            f"  Language   : {self.language.title()}",
            f"  Model      : {self.model_key}",
            "=" * 70,
            "",
        ]

        success_count = sum(1 for e in self.entries if e["status"] == "success")
        skip_count = sum(1 for e in self.entries if e["status"] == "skipped")
        fail_count = sum(1 for e in self.entries if e["status"] == "failed")

        lines += [
            f"  RESULTS: {success_count} transcribed  |  {skip_count} skipped  |  {fail_count} failed",
            "",
        ]

        for entry in self.entries:
            icon = {"success": "✓", "skipped": "–", "failed": "✗"}.get(entry["status"], "?")
            dur = f"  [{entry['duration_s']:.1f}s]" if entry["duration_s"] else ""
            lines.append(f"  {icon}  {entry['video']}{dur}")
            if entry["error"]:
                lines.append(f"      ERROR: {entry['error']}")

        lines += ["", "=" * 70]
        output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core processing loop
# ---------------------------------------------------------------------------

def process_videos(
    videos: List[Path],
    engine: Transcriber,
    force: bool,
    write_log: bool,
    folder: Path,
    language: str,
    model_key: str,
) -> tuple[int, int, int]:
    """
    Transcribe all videos, showing a rich progress UI.

    Returns:
        Tuple of (success_count, skipped_count, failed_count).
    """
    log = TranscriptionLog(folder, language, model_key) if write_log else None

    success = 0
    skipped = 0
    failed = 0

    overall_progress = make_overall_progress()
    video_progress = make_video_progress()

    with overall_progress, video_progress:
        overall_task = overall_progress.add_task(
            "Transcribing videos", total=len(videos)
        )

        for video in videos:
            video_name = video.name
            srt_out = subtitle_path(video)

            # Skip if already done (unless --force)
            if not force and not needs_transcription(video):
                overall_progress.update(
                    overall_task,
                    advance=1,
                    description=f"[dim]Skipped: {video_name}[/]",
                )
                skipped += 1
                if log:
                    log.record(video, "skipped")
                continue

            # Update the overall bar description
            overall_progress.update(
                overall_task,
                description=f"{video_name}",
            )

            # Start per-video spinner
            vid_task = video_progress.add_task(
                f"Extracting & transcribing audio…", total=None
            )

            t_start = time.monotonic()
            error_msg: Optional[str] = None

            try:
                segments = engine.transcribe(video)
                write_srt(segments, srt_out)
                duration = time.monotonic() - t_start
                success += 1

                video_progress.update(
                    vid_task,
                    description=f"[success]Done[/] → {srt_out.name}  "
                    f"[dim]({len(segments)} segments, {duration:.1f}s)[/]",
                )

                if log:
                    log.record(video, "success", duration_s=duration)

            except Exception as exc:
                duration = time.monotonic() - t_start
                error_msg = str(exc)
                failed += 1

                video_progress.update(
                    vid_task,
                    description=f"[error]Failed[/] {video_name}: {error_msg[:60]}",
                )

                if log:
                    log.record(video, "failed", duration_s=duration, error=error_msg)

                # Print full traceback for debugging
                console.print_exception(max_frames=5)

            finally:
                video_progress.stop_task(vid_task)
                overall_progress.advance(overall_task)

    # Write log file
    if log:
        log_path = folder / "transcription_log.txt"
        try:
            log.write(log_path)
            console.print(f"\n[info]📋 Log saved to: [file]{log_path}[/][/]")
        except OSError as e:
            console.print(f"[warning]⚠  Could not write log file: {e}[/]")

    return success, skipped, failed


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_summary(success: int, skipped: int, failed: int, elapsed: float) -> None:
    """Print a final results summary panel."""
    console.print()
    console.print(Rule("[bold cyan]Transcription Complete[/]"))
    console.print()

    cols = []

    success_panel = Panel(
        Text(str(success), style="bold green", justify="center"),
        title="[green]✓ Transcribed[/]",
        border_style="green",
        width=20,
    )
    skipped_panel = Panel(
        Text(str(skipped), style="bold yellow", justify="center"),
        title="[yellow]– Skipped[/]",
        border_style="yellow",
        width=20,
    )
    failed_panel = Panel(
        Text(str(failed), style="bold red", justify="center"),
        title="[red]✗ Failed[/]",
        border_style="red",
        width=20,
    )

    cols = Columns(
        [success_panel, skipped_panel, failed_panel],
        equal=True,
        expand=False,
    )
    console.print(cols, justify="center")
    console.print()

    mins, secs = divmod(int(elapsed), 60)
    hours, mins = divmod(mins, 60)
    time_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
    console.print(f"[dim]  ⏱  Total time: {time_str}[/]")
    console.print()

    if failed == 0 and success > 0:
        console.print(
            Panel(
                "[bold green]All videos transcribed successfully! 🎉[/]",
                border_style="green",
                padding=(0, 4),
            ),
            justify="center",
        )
    elif failed > 0:
        console.print(
            Panel(
                f"[yellow]{failed} video(s) failed to transcribe.[/] "
                f"Check errors above or enable --log for details.",
                border_style="yellow",
                padding=(0, 2),
            ),
            justify="center",
        )
    console.print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description=(
            "Video → Subtitles: Recursively transcribe videos to SRT subtitles\n"
            "using MLX-Whisper (Apple Silicon optimized). Supports English and Arabic."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe videos that already have an .srt file.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Save a transcription_log.txt file in the target folder.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Target folder path (skips the interactive folder prompt).",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["english", "arabic"],
        default=None,
        help="Video language (skips the interactive language prompt).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(SPEED_MODES.keys()),
        default=None,
        help="Speed mode: quality | balanced | fast  (skips interactive prompt).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _apply_speed_mode(mode: dict) -> Optional[subprocess.Popen]:
    """
    Apply OS-level optimisations for the chosen speed mode.

    - Sets MLX_GPU_MEMORY_LIMIT so MLX can use more of the 32 GB unified pool.
    - Sets PYTORCH_ENABLE_MPS_FALLBACK to keep ops on-chip.
    - Optionally starts `caffeinate` to prevent macOS from throttling the process.

    Returns the caffeinate Popen handle (so caller can kill it on exit), or None.
    """
    if sys.platform != "darwin":
        return None

    # Environment variables must be set BEFORE mlx is imported.
    os.environ["MLX_GPU_MEMORY_LIMIT"] = mode["gpu_mem_limit"]
    if mode["mps_fallback"]:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    caffeinate_proc: Optional[subprocess.Popen] = None
    if mode["caffeinate"]:
        try:
            # -i = prevent idle sleep, -s = prevent system sleep
            caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-i", "-s"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            console.print(
                f"  [success]✓[/] caffeinate started [dim](PID {caffeinate_proc.pid})[/]"
            )
        except FileNotFoundError:
            console.print("  [warning]⚠  caffeinate not found — skipping[/]")

    return caffeinate_proc


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print_banner()

    # ── Interactive prompts (skipped if CLI args supplied) ──────────────────

    folder: Path = (
        Path(args.folder).expanduser().resolve()
        if args.folder
        else prompt_folder()
    )

    language: str = args.language if args.language else prompt_language()
    speed_mode: dict = SPEED_MODES[args.mode] if args.mode else prompt_speed_mode()
    model_key: str = speed_mode["model"]

    # ── Scan for videos ─────────────────────────────────────────────────────

    console.print(Rule("[secondary]Scanning for videos…[/]"))
    console.print()

    with console.status("[cyan]Scanning directory tree…[/]", spinner="dots"):
        try:
            all_videos = scan_videos(folder)
        except NotADirectoryError as e:
            console.print(f"[error]✗  {e}[/]")
            sys.exit(1)

    if not all_videos:
        console.print(
            Panel(
                f"[yellow]No video files found in:[/]\n[file]{folder}[/]",
                border_style="yellow",
                title="[yellow]No Videos Found[/]",
                padding=(1, 4),
            )
        )
        sys.exit(0)

    # Separate into "needs work" vs "already done"
    to_process = [v for v in all_videos if args.force or needs_transcription(v)]
    skipped_existing = len(all_videos) - len(to_process)

    console.print(
        f"  [success]✓[/] Found [bold]{len(all_videos)}[/] video(s) "
        f"([bold]{len(to_process)}[/] to transcribe, "
        f"[dim]{skipped_existing} already done[/])"
    )
    console.print()

    # ── Confirm ─────────────────────────────────────────────────────────────

    if not confirm_start(
        folder, language, speed_mode, len(all_videos), len(to_process), skipped_existing
    ):
        console.print("[dim]Exiting.[/]")
        sys.exit(0)

    # ── Apply speed-mode OS optimisations ───────────────────────────────────
    #    Must happen before engine.load() so env vars are visible to MLX.

    console.print(Rule("[secondary]Applying Optimisations[/]"))
    console.print()
    if sys.platform == "darwin":
        console.print(
            f"  [dim]MLX GPU memory limit → [bold]{int(float(speed_mode['gpu_mem_limit'])*100)}%[/] of 32 GB[/]"
        )
        if speed_mode["mps_fallback"]:
            console.print("  [dim]MPS fallback        → enabled[/]")
    else:
        console.print("  [dim]Using CUDA acceleration (faster-whisper)[/]")

    caffeinate_proc = _apply_speed_mode(speed_mode)
    console.print()

    # ── Load model ──────────────────────────────────────────────────────────

    console.print(Rule("[secondary]Loading Model[/]"))
    console.print()

    engine = Transcriber(model_key=model_key, language=language)

    with console.status(
        f"[cyan]Downloading & loading [bold]{model_key}[/] model "
        f"(first run may take a minute)…[/]",
        spinner="dots12",
    ):
        try:
            engine.load()
        except RuntimeError as e:
            console.print(f"\n[error]✗  {e}[/]")
            if caffeinate_proc:
                caffeinate_proc.terminate()
            sys.exit(1)

    console.print(f"  [success]✓[/] Model ready: [accent]{model_key}[/]")
    console.print()

    # ── Transcribe ──────────────────────────────────────────────────────────

    console.print(Rule("[secondary]Transcribing[/]"))
    console.print()

    wall_start = time.monotonic()

    success, skipped, failed = process_videos(
        videos=to_process,
        engine=engine,
        force=args.force,
        write_log=args.log,
        folder=folder,
        language=language,
        model_key=model_key,
    )

    elapsed = time.monotonic() - wall_start

    # Stop caffeinate now that transcription is done
    if caffeinate_proc and caffeinate_proc.poll() is None:
        caffeinate_proc.terminate()

    # Count truly-skipped (not-in-to_process) separately
    total_skipped = skipped + skipped_existing

    print_summary(success, total_skipped, failed, elapsed)


if __name__ == "__main__":
    _caffeinate: Optional[subprocess.Popen] = None
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[warning]⚠  Interrupted by user (Ctrl+C). Exiting.[/]\n")
        sys.exit(130)
    finally:
        # Safety net: always clean up caffeinate if process is still alive
        if _caffeinate and _caffeinate.poll() is None:
            _caffeinate.terminate()
