"""Video reading utilities: metadata extraction, frame seeking, and streaming.

This module provides low-level video access using OpenCV (cv2):
- Extracts container metadata (fps, frame count, resolution, duration)
- Seeks to exact timestamps (millisecond precision)
- Streams frames on-demand to maintain minimal memory footprint (under 4 GB constraint)
- Generates SHA-256 hashes for deduplication and caching
"""

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from videosdk.errors import InvalidVideoError, VideoNotFoundError
from videosdk.types import VideoMeta


def compute_file_hash(path: str | Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 cryptographic hash of a file for deduplication.

    Reads the file in streaming binary chunks (default 64 KB) so very large
    videos never exhaust system RAM.

    Args:
        path: Path to the target video file.
        chunk_size: Byte size read per iteration. Defaults to 64 KB.

    Returns:
        Hexadecimal SHA-256 digest string.

    Raises:
        VideoNotFoundError: If the file does not exist on disk.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise VideoNotFoundError(str(path))

    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_video_metadata(path: str | Path) -> VideoMeta:
    """Extract resolution, frame rate, total frame count, and duration from video.

    Opens the video with OpenCV's VideoCapture, queries container properties,
    and calculates duration. Raises structured errors if the file is missing or invalid.

    Args:
        path: Path to the video file.

    Returns:
        VideoMeta dataclass with video details.

    Raises:
        VideoNotFoundError: If the file does not exist.
        InvalidVideoError: If OpenCV cannot open or decode the video header.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise VideoNotFoundError(str(path))

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise InvalidVideoError(str(path), reason="Could not open video stream with OpenCV")

    try:
        # Read container properties
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check for unreadable / corrupted streams (e.g., zero fps or zero frames)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise InvalidVideoError(
                str(path),
                reason=f"Invalid stream dimensions or fps (fps={fps}, frames={frame_count})",
            )

        duration = frame_count / fps

        return VideoMeta(
            path=str(file_path),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration=duration,
        )
    finally:
        # Always release the OpenCV VideoCapture resource to prevent file descriptor leaks
        cap.release()


def read_frame_at_time(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray:
    """Seek to a specific timestamp in seconds and return the decoded RGB frame.

    Uses millisecond seeking via `cv2.CAP_PROP_POS_MSEC` for precise temporal positioning.
    Converts OpenCV's default BGR color space to RGB because downstream vision models
    (such as CLIP) expect RGB images.

    Args:
        cap: An already opened cv2.VideoCapture instance.
        time_sec: Desired timestamp in seconds (must be non-negative).

    Returns:
        RGB image frame as a NumPy array of shape (height, width, 3) and dtype uint8.

    Raises:
        InvalidVideoError: If seeking or reading the frame at this timestamp fails.
    """
    if not cap.isOpened():
        raise InvalidVideoError("VideoCapture", reason="VideoCapture instance is not opened")

    # Clamp timestamp to at least 0.0 seconds
    target_sec = max(0.0, float(time_sec))

    # cv2.CAP_PROP_POS_MSEC expects milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)

    ret, frame = cap.read()
    if not ret or frame is None:
        raise InvalidVideoError(
            "VideoCapture",
            reason=f"Failed to read frame at timestamp {target_sec:.2f}s",
        )

    # Convert OpenCV default BGR to standard RGB for CLIP / PIL compatibility
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def iter_frames_at_times(
    path: str | Path,
    times: Sequence[float],
) -> Iterator[np.ndarray]:
    """Generator that yields RGB frames at specified timestamps.

    Opens the video stream once, sequentially seeks and yields one frame at a time,
    and automatically releases the capture when done or if an error occurs.
    This guarantees that only one uncompressed frame resides in memory at any point.

    Args:
        path: Path to the video file.
        times: A sequence of timestamps (in seconds) to extract frames for.

    Yields:
        RGB NumPy array for each requested timestamp.

    Raises:
        VideoNotFoundError: If the video file is not found.
        InvalidVideoError: If OpenCV fails to open the video or decode requested frames.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise VideoNotFoundError(str(path))

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise InvalidVideoError(str(path), reason="Could not open video stream with OpenCV")

    try:
        for t in times:
            yield read_frame_at_time(cap, t)
    finally:
        cap.release()
