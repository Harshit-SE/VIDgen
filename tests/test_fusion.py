"""Unit tests for Reciprocal Rank Fusion and Searcher in videosdk.search."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from videosdk.config import Config
from videosdk.search.fusion import reciprocal_rank_fusion
from videosdk.search.searcher import Searcher, format_timestamp
from videosdk.store.metadata import MetadataStore
from videosdk.store.vectors import VectorStore


def test_rrf_basic() -> None:
    """Test standard fusion with overlapping candidates from two sources."""
    # List A (e.g. Visual): item 1 rank 1, item 2 rank 2, item 3 rank 3
    list_a = [(1, 0.9), (2, 0.8), (3, 0.7)]
    # List B (e.g. Speech): item 2 rank 1, item 3 rank 2, item 1 rank 3
    list_b = [(2, 0.95), (3, 0.85), (1, 0.6)]

    result = reciprocal_rank_fusion([list_a, list_b], k=60)

    # Item 2 is rank 2 in A and rank 1 in B:
    # score(2) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # score(1) = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    # score(3) = 1/(60+3) + 1/(60+2) = 1/63 + 1/62 = 0.015873 + 0.016129 = 0.032002
    # Therefore: item 2 must be ranked first, followed by item 1, then item 3.
    assert result[0][0] == 2
    assert result[1][0] == 1
    assert result[2][0] == 3

    expected_score_2 = (1.0 / 62) + (1.0 / 61)
    assert pytest.approx(result[0][1], rel=1e-5) == expected_score_2


def test_rrf_single_list() -> None:
    """Test fusion when only a single modality provides results."""
    list_a = [(5, 0.9), (3, 0.5)]
    result = reciprocal_rank_fusion([list_a], k=60)

    assert len(result) == 2
    assert result[0][0] == 5
    assert result[1][0] == 3
    assert pytest.approx(result[0][1], rel=1e-5) == 1.0 / 61
    assert pytest.approx(result[1][1], rel=1e-5) == 1.0 / 62


def test_rrf_empty() -> None:
    """Test handling of completely empty inputs."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_no_overlap() -> None:
    """Test fusion when the two ranked lists contain disjoint candidates."""
    list_a = [(1, 0.9), (2, 0.8)]
    list_b = [(3, 0.95), (4, 0.85)]

    result = reciprocal_rank_fusion([list_a, list_b], k=60)

    assert len(result) == 4
    # Rank 1 items from each list have score 1/61: items 1 and 3
    # Rank 2 items from each list have score 1/62: items 2 and 4
    top_ids = {result[0][0], result[1][0]}
    bottom_ids = {result[2][0], result[3][0]}
    assert top_ids == {1, 3}
    assert bottom_ids == {2, 4}


def test_rrf_custom_k() -> None:
    """Test that custom k smoothing values work accurately."""
    list_a = [(10, 0.9)]
    result = reciprocal_rank_fusion([list_a], k=10)

    assert result[0][0] == 10
    # rank 1 with k=10 -> 1 / (10 + 1) = 1 / 11
    assert pytest.approx(result[0][1], rel=1e-5) == 1.0 / 11


def test_format_timestamp() -> None:
    """Test converting seconds into MM:SS and H:MM:SS."""
    assert format_timestamp(0.0) == "00:00"
    assert format_timestamp(5.4) == "00:05"
    assert format_timestamp(59.9) == "01:00"
    assert format_timestamp(75.0) == "01:15"
    assert format_timestamp(3600.0) == "1:00:00"
    assert format_timestamp(3665.0) == "1:01:05"


@pytest.fixture
def mock_stores(tmp_path: Path) -> tuple[VectorStore, VectorStore, MetadataStore]:
    """Provide initialized mock vector stores and real SQLite metadata store."""
    v_store = VectorStore(tmp_path / "visual.faiss", dim=512)
    t_store = VectorStore(tmp_path / "text.faiss", dim=384)
    m_store = MetadataStore(tmp_path / "meta.db")
    m_store.create_tables()

    # Seed one test video and three segments
    vid_id = m_store.add_video("test_video.mp4", "hash123", 30.0, 30.0, 1920, 1080)
    m_store.add_segments(
        vid_id,
        [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "transcript_text": "hello and welcome to the show",
                "visual_index_id": 0,
                "text_index_id": 0,
            },
            {
                "start_time": 10.0,
                "end_time": 20.0,
                "transcript_text": "now we discuss neural networks",
                "visual_index_id": 1,
                "text_index_id": 1,
            },
            {
                "start_time": 20.0,
                "end_time": 30.0,
                "transcript_text": "",
                "visual_index_id": 2,
                "text_index_id": -1,
            },
        ],
    )
    return v_store, t_store, m_store


def test_searcher_validation_and_empty_query(
    mock_stores: tuple[VectorStore, VectorStore, MetadataStore],
) -> None:
    """Test query input validation and invalid modes."""
    v_store, t_store, m_store = mock_stores
    searcher = Searcher(v_store, t_store, m_store)

    # Empty query should return empty list
    assert searcher.search("") == []
    assert searcher.search("   ") == []

    # Invalid mode should raise ValueError
    with pytest.raises(ValueError, match="Unsupported search mode"):
        searcher.search("query", mode="invalid_mode")


def test_searcher_modes_and_fusion(
    mock_stores: tuple[VectorStore, VectorStore, MetadataStore],
) -> None:
    """Test visual, speech, and hybrid both modes with mocked candidate lists."""
    v_store, t_store, m_store = mock_stores
    searcher = Searcher(v_store, t_store, m_store, config=Config())

    # Mock the internal encoder search methods
    # Visual candidates: segment 1 rank 1 (score 0.88), segment 2 rank 2 (score 0.75)
    searcher._visual_search = MagicMock(return_value=[(1, 0.88), (2, 0.75)])
    # Speech candidates: segment 2 rank 1 (score 0.92), segment 3 rank 2 (score 0.60)
    searcher._speech_search = MagicMock(return_value=[(2, 0.92), (3, 0.60)])

    # Test Visual Mode
    res_visual = searcher.search("dog", mode="visual", top_k=2)
    assert len(res_visual) == 2
    assert res_visual[0].start == 0.0
    assert res_visual[0].source == "visual"
    assert res_visual[0].score == 0.88

    # Test Speech Mode
    res_speech = searcher.search("neural networks", mode="speech", top_k=2)
    assert len(res_speech) == 2
    assert res_speech[0].start == 10.0
    assert res_speech[0].source == "speech"
    assert res_speech[0].score == 0.92

    # Test Both (Hybrid) Mode:
    # Segment 2 is rank 2 in visual and rank 1 in speech -> highest RRF score
    res_both = searcher.search("neural networks in action", mode="both", top_k=3)
    assert len(res_both) == 3
    # First result should be segment 2 (appeared in both)
    assert res_both[0].start == 10.0
    assert res_both[0].source == "both"
    assert res_both[0].text == "now we discuss neural networks"
    assert res_both[0].timestamp == "00:10"
