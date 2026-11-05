"""Smoke tests to verify the package imports and basic types work."""

import pytest

from videosdk import __version__
from videosdk.config import Config
from videosdk.errors import IndexVersionError, VideoNotFoundError
from videosdk.types import SearchResult


def test_version_exists() -> None:
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_search_result_creation() -> None:
    r = SearchResult(
        video="test.mp4",
        start=10.0,
        end=15.0,
        timestamp="0:10",
        score=0.85,
        text="hello world",
        source="speech",
    )
    assert r.video == "test.mp4"
    assert r.start == 10.0
    assert r.source == "speech"


def test_config_defaults() -> None:
    c = Config()
    assert c.clip_model_name == "ViT-B-32"
    assert c.schema_version == 1


def test_errors() -> None:
    with pytest.raises(VideoNotFoundError, match=r"test\.mp4"):
        raise VideoNotFoundError("test.mp4")
    with pytest.raises(IndexVersionError, match=r"v2.*v1"):
        raise IndexVersionError(found=2, expected=1)
