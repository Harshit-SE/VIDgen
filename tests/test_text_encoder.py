"""Unit tests for sentence-transformers text encoder in videosdk.models.text_encoder."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from videosdk.config import Config
from videosdk.models.text_encoder import TextEncoder, embed_transcripts


def test_text_encoder_init() -> None:
    """Test TextEncoder initialization with default and custom config."""
    cfg = Config(text_dim=384)
    encoder = TextEncoder(cfg)
    assert encoder.config.text_dim == 384
    assert encoder.model is None


def test_text_encoder_empty_encode() -> None:
    """Test encoding an empty list of strings returns shape (0, 384)."""
    encoder = TextEncoder()
    res = encoder.encode([])
    assert isinstance(res, np.ndarray)
    assert res.shape == (0, 384)
    assert res.dtype == np.float32


def test_text_encoder_missing_package() -> None:
    """Test that missing sentence_transformers package raises actionable ImportError."""
    encoder = TextEncoder()
    with (
        patch.dict(sys.modules, {"sentence_transformers": None}),
        pytest.raises(ImportError, match="sentence-transformers is required"),
    ):
        encoder.load()


def test_text_encoder_mocked_encode() -> None:
    """Test encoding text strings with a mocked SentenceTransformer."""
    mock_st_cls = MagicMock()
    mock_model_instance = MagicMock()

    # Create dummy embeddings (2 sentences, 384 dim)
    dummy_vecs = np.zeros((2, 384), dtype=np.float32)
    dummy_vecs[0, 0] = 1.0
    dummy_vecs[1, 1] = 1.0

    mock_model_instance.encode.return_value = dummy_vecs
    mock_st_cls.return_value = mock_model_instance

    mock_module = ModuleType("sentence_transformers")
    mock_module.SentenceTransformer = mock_st_cls  # type: ignore[attr-defined]

    encoder = TextEncoder()
    with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
        encoder.load()
        mock_st_cls.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")

        texts = ["first sentence", "second sentence"]
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32
        assert embeddings[0, 0] == 1.0

        # Test single query encoding
        query_vec = encoder.encode_query("search query")
        assert query_vec.shape == (384,)
        assert query_vec.dtype == np.float32

        # Test unload
        encoder.unload()
        assert encoder.model is None


def test_embed_transcripts_helper() -> None:
    """Test embed_transcripts lifecycle helper."""
    dummy_vecs = np.ones((1, 384), dtype=np.float32)

    with (
        patch.object(TextEncoder, "load") as mock_load,
        patch.object(TextEncoder, "encode", return_value=dummy_vecs) as mock_encode,
        patch.object(TextEncoder, "unload") as mock_unload,
    ):
        res = embed_transcripts(["sample transcript"])
        assert res.shape == (1, 384)
        mock_load.assert_called_once()
        mock_encode.assert_called_once_with(["sample transcript"])
        mock_unload.assert_called_once()
