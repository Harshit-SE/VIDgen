"""Scene detection and temporal video segmentation using OpenCV HSV histograms.

This module splits a continuous video stream into semantically distinct segments:
1. Samples frames at fixed intervals (default: 0.5s) to save compute.
2. Converts frames to HSV color space and computes 2D (Hue-Saturation) histograms.
3. Compares consecutive histograms using correlation (cv2.HISTCMP_CORREL).
4. Drops below threshold (default: 0.7) signify shot/scene boundaries.
5. Applies segment rules:
   - Merges segments shorter than min_seconds (default: 2.0s).
   - Splits segments longer than max_seconds (default: 30.0s) into ~10s windows.
   - Falls back to fixed windows if scene count is suspiciously low (< duration / 30).
"""

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from videosdk.config import Config
from videosdk.errors import InvalidVideoError, VideoNotFoundError
from videosdk.video.reader import get_video_metadata, read_frame_at_time


def _compute_histograms(
    path: str | Path,
    sample_interval: float = 0.5,
) -> list[tuple[float, np.ndarray]]:
    """Sample frames at regular intervals and calculate normalized HSV histograms.

    Args:
        path: Video file path.
        sample_interval: Time in seconds between sampled frames.

    Returns:
        List of (timestamp_seconds, normalized_histogram) tuples.

    Raises:
        VideoNotFoundError: If the video does not exist.
        InvalidVideoError: If OpenCV fails to open the video.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise VideoNotFoundError(str(path))

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise InvalidVideoError(str(path), reason="Could not open video stream for scene detection")

    histograms: list[tuple[float, np.ndarray]] = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise InvalidVideoError(str(path), reason="Video has invalid fps or frame count")

        duration = frame_count / fps
        current_time = 0.0

        while current_time < duration:
            try:
                frame_rgb = read_frame_at_time(cap, current_time)
            except InvalidVideoError:
                break

            # Convert RGB frame to HSV (Hue, Saturation, Value)
            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

            # Compute 2D histogram on H (50 bins) and S (60 bins)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

            # Normalize histogram to [0, 1] range
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            histograms.append((current_time, hist))
            current_time += sample_interval

    finally:
        cap.release()

    return histograms


def _find_boundaries(
    histograms: Sequence[tuple[float, np.ndarray]],
    threshold: float = 0.7,
) -> list[float]:
    """Identify boundary timestamps where histogram correlation drops below threshold.

    Args:
        histograms: Sequence of (timestamp, hist_matrix) tuples.
        threshold: Correlation coefficient below which a cut is detected (0.0 to 1.0).

    Returns:
        List of scene boundary timestamps in ascending order.
    """
    boundaries: list[float] = []
    if len(histograms) < 2:
        return boundaries

    for i in range(1, len(histograms)):
        _time_prev, hist_prev = histograms[i - 1]
        time_curr, hist_curr = histograms[i]

        # cv2.HISTCMP_CORREL returns 1.0 for perfect match, -1.0 for complete mismatch
        correlation = float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL))

        if math.isnan(correlation) or correlation < threshold:
            # Mark the boundary at the second timestamp where change occurred
            boundaries.append(time_curr)

    return boundaries


def _apply_segment_rules(
    boundaries: Sequence[float],
    duration: float,
    config: Config,
) -> list[tuple[float, float]]:
    """Convert boundaries into valid segments enforcing duration bounds.

    Rules applied:
    1. If detected scenes are too sparse (< duration / 30), fall back to fixed windows.
    2. Convert boundaries into (start, end) intervals covering [0, duration].
    3. Merge short segments (< min_seconds) with the shorter adjacent neighbor.
    4. Split long segments (> max_seconds) into equal chunks of ~fallback_seconds.

    Args:
        boundaries: Raw detected boundary cut timestamps.
        duration: Total video length in seconds.
        config: Configuration containing threshold and duration limits.

    Returns:
        List of (start, end) segment tuples rounded to 2 decimal places.
    """
    if duration <= 0:
        return []

    # 1. Fallback check: static videos (podcasts, slide decks) have almost no cuts
    min_expected_scenes = duration / 30.0
    # Initial raw segments count is boundaries count + 1
    if (len(boundaries) + 1) < min_expected_scenes:
        return _generate_fixed_windows(duration, config.segment_fallback_seconds)

    # 2. Build initial segments between cuts
    sorted_cuts = sorted({b for b in boundaries if 0.0 < b < duration})
    points = [0.0, *sorted_cuts, duration]
    segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]

    # 3. Merge pass: merge segments shorter than config.segment_min_seconds
    merged_segments: list[tuple[float, float]] = []
    i = 0
    while i < len(segments):
        start, end = segments[i]
        seg_dur = end - start

        if seg_dur < config.segment_min_seconds:
            if not merged_segments:
                # First segment: merge into next if available
                if i + 1 < len(segments):
                    _, next_end = segments[i + 1]
                    segments[i + 1] = (start, next_end)
                else:
                    merged_segments.append((start, end))
            else:
                # Merge into previous segment
                prev_start, _ = merged_segments[-1]
                merged_segments[-1] = (prev_start, end)
        else:
            merged_segments.append((start, end))
        i += 1

    if not merged_segments:
        merged_segments = [(0.0, duration)]

    # 4. Split pass: split segments longer than config.segment_max_seconds
    final_segments: list[tuple[float, float]] = []
    for start, end in merged_segments:
        seg_dur = end - start
        if seg_dur > config.segment_max_seconds:
            # Divide into n equal parts approximating fallback_seconds
            n_parts = max(2, math.ceil(seg_dur / config.segment_fallback_seconds))
            part_dur = seg_dur / n_parts
            for p in range(n_parts):
                p_start = start + p * part_dur
                p_end = start + (p + 1) * part_dur if p < n_parts - 1 else end
                final_segments.append((round(p_start, 2), round(p_end, 2)))
        else:
            final_segments.append((round(start, 2), round(end, 2)))

    return final_segments


def _generate_fixed_windows(
    duration: float,
    window_size: float = 10.0,
) -> list[tuple[float, float]]:
    """Generate fixed-interval sliding windows across the duration."""
    windows: list[tuple[float, float]] = []
    t = 0.0
    while t < duration:
        end = min(duration, t + window_size)
        # If the remaining trailing segment is tiny (< 2s) and there is a prior window, extend prior
        if (end - t) < 2.0 and windows:
            prev_start, _ = windows[-1]
            windows[-1] = (prev_start, round(end, 2))
            break
        windows.append((round(t, 2), round(end, 2)))
        t += window_size

    return windows


def detect_scenes(
    path: str | Path,
    config: Config | None = None,
) -> list[tuple[float, float]]:
    """Detect scenes in a video and return normalized (start, end) timestamps.

    Args:
        path: Path to the video file.
        config: Optional configuration instance. If omitted, default Config() is used.

    Returns:
        List of (start_sec, end_sec) tuples covering the full video timeline.
    """
    cfg = config or Config()
    meta = get_video_metadata(path)

    # If video is shorter than minimum segment duration, return single segment
    if meta.duration <= cfg.segment_min_seconds:
        return [(0.0, round(meta.duration, 2))]

    histograms = _compute_histograms(path, sample_interval=cfg.scene_sample_interval)
    boundaries = _find_boundaries(histograms, threshold=cfg.scene_hist_threshold)
    return _apply_segment_rules(boundaries, meta.duration, cfg)
