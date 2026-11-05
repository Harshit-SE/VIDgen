"""Video clipping utility using FFmpeg."""

import shutil
import subprocess
from pathlib import Path

from videosdk.errors import FFmpegNotFoundError, VideoNotFoundError


def cut_clip(video_path: str, start: float, end: float, output_path: str) -> str:
    """Cut a clip from a video using FFmpeg stream copy.

    Args:
        video_path: Path to the source video.
        start: Start timestamp in seconds.
        end: End timestamp in seconds.
        output_path: Path where the clipped video will be saved.

    Returns:
        The output path string.
    """
    if not Path(video_path).exists():
        raise VideoNotFoundError(video_path)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FFmpegNotFoundError()

    duration = max(0.0, end - start)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-c",
        "copy",
        output_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg clipping failed: {result.stderr}")

    return output_path
