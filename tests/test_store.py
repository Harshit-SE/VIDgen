"""Unit tests for FAISS vector storage and SQLite metadata storage."""

from pathlib import Path

import numpy as np
import pytest

from videosdk.store.metadata import MetadataStore
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


def test_metadata_store_crud(tmp_path: Path) -> None:
    """Test creating tables, adding videos, adding segments, and querying them."""
    db_path = tmp_path / "meta.sqlite"
    with MetadataStore(db_path) as store:
        store.create_tables()

        # Add video
        video_id = store.add_video(
            file_path="lecture.mp4",
            file_hash="hash_12345",
            duration=120.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        assert video_id == 1

        # Query video
        v_by_path = store.get_video("lecture.mp4")
        assert v_by_path is not None
        assert v_by_path["file_hash"] == "hash_12345"
        assert v_by_path["duration"] == 120.0

        v_by_id = store.get_video_by_id(video_id)
        assert v_by_id is not None
        assert v_by_id["file_path"] == "lecture.mp4"

        # Check deduplication
        assert store.has_video_hash("hash_12345") is True
        assert store.has_video_hash("unknown_hash") is False

        # Add segments
        segments = [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "transcript_text": "Introduction to neural networks.",
                "visual_index_id": 0,
                "text_index_id": 0,
            },
            {
                "start_time": 10.0,
                "end_time": 25.0,
                "transcript_text": "Here is how backpropagation works.",
                "visual_index_id": 1,
                "text_index_id": 1,
            },
            {
                "start_time": 25.0,
                "end_time": 40.0,
                "transcript_text": "",  # silent segment
                "visual_index_id": 2,
                "text_index_id": -1,
            },
        ]
        store.add_segments(video_id, segments)

        # Query segments for video
        retrieved_segments = store.get_segments_for_video(video_id)
        assert len(retrieved_segments) == 3
        assert retrieved_segments[0]["start_time"] == 0.0
        assert retrieved_segments[0]["transcript_text"] == "Introduction to neural networks."
        assert retrieved_segments[2]["text_index_id"] == -1

        # Query by visual and text indices
        seg_vis = store.get_segment_by_visual_index(1)
        assert seg_vis is not None
        assert seg_vis["start_time"] == 10.0

        seg_text = store.get_segment_by_text_index(0)
        assert seg_text is not None
        assert seg_text["transcript_text"] == "Introduction to neural networks."

        # List videos with segment counts
        v_list = store.list_videos()
        assert len(v_list) == 1
        assert v_list[0]["segment_count"] == 3


def test_metadata_store_cascade_delete(tmp_path: Path) -> None:
    """Test that removing a video automatically cascades and deletes all associated segments."""
    db_path = tmp_path / "meta.sqlite"
    store = MetadataStore(db_path)
    store.create_tables()

    vid = store.add_video(
        file_path="clip.mp4",
        file_hash="hash_clip",
        duration=30.0,
        fps=24.0,
        width=1280,
        height=720,
    )
    store.add_segments(
        vid,
        [
            {
                "start_time": 0.0,
                "end_time": 15.0,
                "transcript_text": "First half",
                "visual_index_id": 0,
                "text_index_id": 0,
            }
        ],
    )
    assert len(store.get_segments_for_video(vid)) == 1

    # Remove video
    deleted = store.remove_video("clip.mp4")
    assert deleted is True
    assert store.get_video("clip.mp4") is None
    assert len(store.get_segments_for_video(vid)) == 0
    store.close()
