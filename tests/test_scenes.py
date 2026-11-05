"""Unit tests for scene detection and segmentation rules in videosdk.video.scenes."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from videosdk.config import Config
from videosdk.video.scenes import (
    _apply_segment_rules,
    _find_boundaries,
    _generate_fixed_windows,
    detect_scenes,
)


@pytest.fixture
def config() -> Config:
    return Config(
        scene_hist_threshold=0.7,
        scene_sample_interval=0.5,
        segment_min_seconds=2.0,
        segment_max_seconds=30.0,
        segment_fallback_seconds=10.0,
    )


def test_find_boundaries_detects_drop(config: Config) -> None:
    """Test that boundary detection marks timestamps where correlation drops below threshold."""
    # Create two histograms with top-half filled and one with bottom-half filled
    h1 = np.zeros((50, 60), dtype=np.float32)
    h1[:25, :] = 1.0
    h2 = np.zeros((50, 60), dtype=np.float32)
    h2[:25, :] = 1.0
    h3 = np.zeros((50, 60), dtype=np.float32)
    h3[25:, :] = 1.0

    histograms = [
        (0.0, h1),
        (0.5, h2),  # identical to h1 -> correlation 1.0
        (1.0, h3),  # opposite to h2 -> correlation -1.0 < 0.7
        (1.5, h3),  # identical to h3 -> correlation 1.0
    ]

    boundaries = _find_boundaries(histograms, threshold=config.scene_hist_threshold)
    assert boundaries == [1.0]


def test_apply_segment_rules_merge_short(config: Config) -> None:
    """Test that segments shorter than 2.0s are merged."""
    # Video duration 20s. Cuts at 1.0s (segment 0-1 is 1s, < min 2s).
    # Since duration is 20s, duration/30 = 0.67. 2 cuts give 3 segments (no fallback).
    boundaries = [1.0, 10.0]
    segments = _apply_segment_rules(boundaries, duration=20.0, config=config)

    # All segments should be >= 2.0s
    for start, end in segments:
        assert (end - start) >= config.segment_min_seconds
    assert segments[0][0] == 0.0
    assert segments[-1][1] == 20.0


def test_apply_segment_rules_split_long(config: Config) -> None:
    """Test that segments longer than 30s are split into ~10s windows."""
    # Duration 120s with 5 boundaries so no fallback
    boundaries = [5.0, 10.0, 15.0, 20.0, 60.0]
    segments = _apply_segment_rules(boundaries, duration=120.0, config=config)

    # No segment should exceed max_seconds
    for start, end in segments:
        dur = round(end - start, 2)
        assert dur <= config.segment_max_seconds
    assert segments[0][0] == 0.0
    assert segments[-1][1] == 120.0


def test_apply_segment_rules_fallback_static_video(config: Config) -> None:
    """Test fallback to fixed 10s windows when scene count < duration / 30."""
    # A 90-second video with 0 detected boundaries (static presentation)
    # Expected scene count for 90s is 90 / 30 = 3 scenes. 1 segment < 3 -> fallback!
    segments = _apply_segment_rules([], duration=90.0, config=config)

    # 90s / 10s = 9 windows
    assert len(segments) == 9
    assert segments[0] == (0.0, 10.0)
    assert segments[-1] == (80.0, 90.0)


def test_generate_fixed_windows() -> None:
    """Test fixed window generator."""
    windows = _generate_fixed_windows(duration=25.0, window_size=10.0)
    assert windows == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]

    # Short trailing remainder (< 2s) merged with prior
    windows_short_remainder = _generate_fixed_windows(duration=21.0, window_size=10.0)
    assert windows_short_remainder == [(0.0, 10.0), (10.0, 21.0)]


def test_detect_scenes_synthetic_video(tmp_path: Path, config: Config) -> None:
    """Create a synthetic 6-second video with an abrupt color change at 3s."""
    video_path = tmp_path / "scenes_test.mp4"
    width, height, fps = 64, 64, 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        # First 3 seconds (30 frames): pure blue
        for _ in range(30):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = 255  # Blue in BGR
            out.write(frame)
        # Next 3 seconds (30 frames): pure red
        for _ in range(30):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 2] = 255  # Red in BGR
            out.write(frame)
    finally:
        out.release()

    segments = detect_scenes(video_path, config)
    assert len(segments) >= 1
    assert segments[0][0] == 0.0
    assert pytest.approx(segments[-1][1], abs=0.2) == 6.0
