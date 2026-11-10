"""Text embedding model wrapper using sentence-transformers (all-MiniLM-L6-v2).

Converts transcript chunks and textual queries into 384-dimensional dense vectors
for semantic search. Always executes on CPU to keep all GPU VRAM available for
visual and speech models.
"""

import gc
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from videosdk.config import Config

logger = logging.getLogger(__name__)


class TextEncoder:
    """Encodes text strings into normalized 384-dim embeddings via sentence-transformers."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize TextEncoder configuration.

        Args:
            config: VideoSDK configuration instance.
        """
        self.config = config or Config()
        self.model: Any = None

    def load(self) -> None:
        """Load the SentenceTransformer model on CPU."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required for transcript embedding. "
                "Install it with: pip install sentence-transformers"
            ) from err

        logger.info(
            "Loading sentence-transformers '%s' on CPU...",
            self.config.text_model_name,
        )
        self.model = SentenceTransformer(
            self.config.text_model_name,
            device="cpu",
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of text strings into L2-normalized 384-dim vectors.

        Args:
            texts: List or sequence of text strings to encode.

        Returns:
            2D numpy array of shape (len(texts), 384) with float32 values.
        """
        if len(texts) == 0:
            return np.empty((0, self.config.text_dim), dtype=np.float32)

        if self.model is None:
            self.load()

        # sentence-transformers supports normalize_embeddings=True directly
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.config.text_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a 1D vector of shape (384,).

        Args:
            query: The search query string.

        Returns:
            1D numpy array of shape (384,) with float32 values.
        """
        result = self.encode([query])
        return np.asarray(result[0], dtype=np.float32)

    def unload(self) -> None:
        """Purge the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()


def embed_transcripts(
    transcripts: Sequence[str],
    config: Config | None = None,
) -> np.ndarray:
    """Convenience helper to encode transcript strings with isolated model lifecycle.

    Args:
        transcripts: Sequence of transcript strings.
        config: VideoSDK configuration instance.

    Returns:
        2D numpy array of shape (len(transcripts), text_dim).
    """
    cfg = config or Config()
    encoder = TextEncoder(cfg)
    encoder.load()
    try:
        return encoder.encode(transcripts)
    finally:
        encoder.unload()
