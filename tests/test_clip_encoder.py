"""Unit tests for OpenCLIP visual and text encoding."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("open_clip")
pytest.importorskip("torch")

from videosdk.config import Config
from videosdk.models.clip_encoder import CLIPEncoder, embed_segments


@pytest.fixture
def clip_encoder() -> CLIPEncoder:
    """Fixture providing a CLIPEncoder instance."""
    encoder = CLIPEncoder()
    encoder.load()
    try:
        yield encoder
    finally:
        encoder.unload()


def test_encode_text(clip_encoder: CLIPEncoder) -> None:
    """Test text query encoding produces normalized 1x512 vector."""
    vec = clip_encoder.encode_text("a dog running in the park")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, 512)
    assert vec.dtype == np.float32

    # Vector must be L2 normalized (unit length)
    norm = float(np.linalg.norm(vec[0]))
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_encode_frames(clip_encoder: CLIPEncoder) -> None:
    """Test frame encoding produces normalized Nx512 matrix."""
    f1 = np.full((64, 64, 3), fill_value=50, dtype=np.uint8)
    f2 = np.full((64, 64, 3), fill_value=200, dtype=np.uint8)

    matrix = clip_encoder.encode_frames([f1, f2])
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 512)
    assert matrix.dtype == np.float32

    for row in matrix:
        norm = float(np.linalg.norm(row))
        assert pytest.approx(norm, rel=1e-3) == 1.0


def test_encode_frames_empty(clip_encoder: CLIPEncoder) -> None:
    """Test encoding empty frame sequence returns (0, 512)."""
    matrix = clip_encoder.encode_frames([])
    assert matrix.shape == (0, 512)


def test_embed_segments(tmp_path: Path) -> None:
    """Test full embed_segments pipeline on synthetic video."""
    import cv2

    video_path = tmp_path / "clip_test.mp4"
    width, height, fps = 64, 64, 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    try:
        for i in range(20):
            frame = np.full((height, width, 3), fill_value=i * 10, dtype=np.uint8)
            out.write(frame)
    finally:
        out.release()

    segments = [(0.0, 1.0), (1.0, 2.0)]
    embeddings = embed_segments(video_path, segments, config=Config())

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 512)
    assert embeddings.dtype == np.float32
    for vec in embeddings:
        norm = float(np.linalg.norm(vec))
        assert pytest.approx(norm, rel=1e-3) == 1.0
