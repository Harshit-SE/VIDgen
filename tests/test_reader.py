"""Unit tests for videosdk.video.reader."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from videosdk.errors import InvalidVideoError, VideoNotFoundError
from videosdk.types import VideoMeta
from videosdk.video.reader import (
    compute_file_hash,
    get_video_metadata,
    iter_frames_at_times,
    read_frame_at_time,
)


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Generate a small 1-second synthetic MP4 video (30 fps, 64x64) for tests."""
    video_path = tmp_path / "test_sample.mp4"
    width, height, fps = 64, 64, 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        for i in range(30):
            # Create a simple colored frame (gradient)
            frame = np.full((height, width, 3), fill_value=i * 8, dtype=np.uint8)
            out.write(frame)
    finally:
        out.release()

    return video_path


def test_compute_file_hash(sample_video: Path) -> None:
    """Test SHA-256 hash calculation."""
    h = compute_file_hash(sample_video)
    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256 hex string length


def test_compute_file_hash_missing() -> None:
    """Test that missing file raises VideoNotFoundError."""
    with pytest.raises(VideoNotFoundError):
        compute_file_hash("nonexistent_video.mp4")


def test_get_video_metadata(sample_video: Path) -> None:
    """Test extracting metadata from a valid video."""
    meta = get_video_metadata(sample_video)
    assert isinstance(meta, VideoMeta)
    assert meta.frame_count == 30
    assert meta.fps == 30.0
    assert meta.width == 64
    assert meta.height == 64
    assert pytest.approx(meta.duration, rel=1e-2) == 1.0


def test_get_video_metadata_missing() -> None:
    """Test get_video_metadata with missing file."""
    with pytest.raises(VideoNotFoundError):
        get_video_metadata("does_not_exist.mp4")


def test_get_video_metadata_invalid(tmp_path: Path) -> None:
    """Test get_video_metadata with an empty or non-video file."""
    fake_video = tmp_path / "corrupt.mp4"
    fake_video.write_text("not a video")
    with pytest.raises(InvalidVideoError):
        get_video_metadata(fake_video)


def test_read_frame_at_time(sample_video: Path) -> None:
    """Test reading a frame at a specific timestamp."""
    cap = cv2.VideoCapture(str(sample_video))
    try:
        frame = read_frame_at_time(cap, 0.5)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8
    finally:
        cap.release()


def test_iter_frames_at_times(sample_video: Path) -> None:
    """Test streaming frames at timestamps."""
    timestamps = [0.1, 0.4, 0.8]
    frames = list(iter_frames_at_times(sample_video, timestamps))
    assert len(frames) == len(timestamps)
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (64, 64, 3)

