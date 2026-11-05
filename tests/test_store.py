"""Unit tests for FAISS vector storage in videosdk.store.vectors."""

from pathlib import Path

import numpy as np
import pytest

from videosdk.store.vectors import VectorStore


def test_vector_store_create_and_add() -> None:
    """Test creating an index and adding vectors."""
    store = VectorStore(path="temp.faiss", dim=4)
    store.create()
    assert store.count() == 0

    # Add 3 vectors of dim=4
    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    start_id = store.add(vecs)
    assert start_id == 0
    assert store.count() == 3


def test_vector_store_search() -> None:
    """Test searching for nearest neighbors via inner product."""
    store = VectorStore(path="temp.faiss", dim=4)
    store.create()

    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # id 0: perfectly aligns with query
            [0.0, 1.0, 0.0, 0.0],  # id 1: orthogonal
            [0.7071, 0.7071, 0.0, 0.0],  # id 2: partial alignment
        ],
        dtype=np.float32,
    )
    store.add(vecs)

    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scores, indices = store.search(query, top_k=2)

    assert len(scores) == 2
    assert len(indices) == 2
    # Best match should be id 0 with score 1.0
    assert indices[0] == 0
    assert pytest.approx(scores[0], rel=1e-3) == 1.0
    # Second match should be id 2 with score ~0.7071
    assert indices[1] == 2
    assert pytest.approx(scores[1], rel=1e-3) == 0.7071


def test_vector_store_save_and_load(tmp_path: Path) -> None:
    """Test persisting and re-loading index from disk."""
    index_path = tmp_path / "test.faiss"
    store1 = VectorStore(path=index_path, dim=4)
    store1.create()

    vecs = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    store1.add(vecs)
    store1.save()

    assert index_path.is_file()

    # Load in new store
    store2 = VectorStore(path=index_path, dim=4)
    store2.load()
    assert store2.count() == 1

    scores, indices = store2.search(vecs[0], top_k=1)
    assert indices[0] == 0
    assert pytest.approx(scores[0], rel=1e-3) == 1.0


def test_vector_store_dimension_mismatch() -> None:
    """Test that mismatched vector dimensions raise ValueError."""
    store = VectorStore(path="temp.faiss", dim=4)
    store.create()

    wrong_dim_vec = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        store.add(wrong_dim_vec)

    with pytest.raises(ValueError, match="Query dimension mismatch"):
        store.search(wrong_dim_vec)

