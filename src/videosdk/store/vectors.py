"""Vector storage and exact similarity search using FAISS.

Stores L2-normalized embeddings using FAISS IndexFlatIP (Inner Product).
Because vectors are unit-length (L2 norm = 1.0), the inner product is mathematically
equivalent to cosine similarity:
    cos(u, v) = (u · v) / (||u|| * ||v||) = u · v
"""

from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorStore:
    """Manages an on-disk FAISS index for high-speed similarity search."""

    def __init__(self, path: str | Path, dim: int = 512) -> None:
        """Initialize VectorStore.

        Args:
            path: Destination file path for saving/loading the .faiss index.
            dim: Vector dimensionality (default: 512 for CLIP ViT-B-32).
        """
        self.path = Path(path)
        self.dim = dim
        self.index: Any = None

    def create(self) -> None:
        """Create a new, empty in-memory FAISS IndexFlatIP."""
        self.index = faiss.IndexFlatIP(self.dim)

    def load(self) -> None:
        """Load an existing FAISS index from disk.

        If the file does not exist, initializes a new empty index.
        """
        if self.path.is_file():
            self.index = faiss.read_index(str(self.path))
            if self.index.d != self.dim:
                raise ValueError(
                    f"Index dimension mismatch: index has {self.index.d}, expected {self.dim}"
                )
        else:
            self.create()

    def add(self, vectors: np.ndarray) -> int:
        """Add vectors to the FAISS index.

        Args:
            vectors: 2D numpy array of shape (N, dim) with float32 values.

        Returns:
            The starting index (row ID) where these vectors were inserted.
        """
        if self.index is None:
            self.create()

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: got {vectors.shape[1]}, expected {self.dim}"
            )

        # FAISS requires contiguous C-order float32 arrays
        contiguous_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        start_id = int(self.index.ntotal)
        self.index.add(contiguous_vectors)
        return start_id

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for the most similar vectors.

        Args:
            query: Query vector of shape (dim,) or (1, dim).
            top_k: Number of nearest neighbors to return.

        Returns:
            Tuple of (scores, indices) as 1D arrays:
            - scores: similarity scores (higher is better, in [-1.0, 1.0])
            - indices: integer segment IDs in the index
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension mismatch: got {query.shape[1]}, expected {self.dim}"
            )

        if self.index is None or self.index.ntotal == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

        contiguous_query = np.ascontiguousarray(query, dtype=np.float32)
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(contiguous_query, k)
        return scores[0], indices[0]

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        if self.index is None:
            self.create()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path))

    def count(self) -> int:
        """Return the total number of vectors in the index."""
        if self.index is None:
            return 0
        return int(self.index.ntotal)
