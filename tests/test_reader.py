"""Unit tests for videosdk.video.reader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from videosdk.errors import FFmpegNotFoundError, InvalidVideoError, VideoNotFoundError
from videosdk.types import VideoMeta
from videosdk.video.reader import (
    compute_file_hash,
    extract_audio,
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
    """Test extracting metadata from synthetic video."""
    meta = get_video_metadata(sample_video)
    assert isinstance(meta, VideoMeta)
    assert meta.fps == 30.0
    assert meta.frame_count == 30
    assert meta.width == 64
    assert meta.height == 64
    assert pytest.approx(meta.duration, rel=1e-2) == 1.0


def test_get_video_metadata_missing() -> None:
    """Test metadata extraction fails on missing file."""
    with pytest.raises(VideoNotFoundError):
        get_video_metadata("missing.mp4")


def test_get_video_metadata_invalid(tmp_path: Path) -> None:
    """Test metadata extraction fails on invalid/corrupted video file."""
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


def test_extract_audio_missing_video() -> None:
    """Test that extract_audio raises VideoNotFoundError for non-existent video."""
    with pytest.raises(VideoNotFoundError):
        extract_audio("nonexistent.mp4", "output.wav")


def test_extract_audio_missing_ffmpeg(sample_video: Path, tmp_path: Path) -> None:
    """Test that extract_audio raises FFmpegNotFoundError when ffmpeg is not installed."""
    out_wav = tmp_path / "audio.wav"
    with patch("shutil.which", return_value=None), pytest.raises(FFmpegNotFoundError):
        extract_audio(sample_video, out_wav)


def test_extract_audio_success(sample_video: Path, tmp_path: Path) -> None:
    """Test that extract_audio executes ffmpeg subprocess command with correct args."""
    out_wav = tmp_path / "audio.wav"
    mock_run = MagicMock(returncode=0)

    with (
        patch("shutil.which", return_value="C:\\ffmpeg\\ffmpeg.exe"),
        patch("subprocess.run", return_value=mock_run) as mock_sp,
    ):
        res = extract_audio(sample_video, out_wav, sample_rate=16000)
        assert res == str(out_wav)
        mock_sp.assert_called_once()
        cmd = mock_sp.call_args[0][0]
        assert cmd[0] == "C:\\ffmpeg\\ffmpeg.exe"
        assert "-vn" in cmd
        assert "16000" in cmd


def test_extract_audio_failure(sample_video: Path, tmp_path: Path) -> None:
    """Test that extract_audio raises RuntimeError when ffmpeg exits with error code."""
    out_wav = tmp_path / "audio.wav"
    mock_run = MagicMock(returncode=1, stderr="Error decoding stream")

    with (
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("subprocess.run", return_value=mock_run),
        pytest.raises(RuntimeError, match="FFmpeg audio extraction failed"),
    ):
        extract_audio(sample_video, out_wav)
