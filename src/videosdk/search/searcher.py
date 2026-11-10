"""Multi-modal search coordinator for VideoSDK.

Orchestrates visual search (CLIP), speech search (Whisper + MiniLM), and unified
hybrid search using Reciprocal Rank Fusion (RRF).

Maintains strict memory safety for 4GB VRAM environments:
- Loads CLIP text encoder only during visual search, then unloads it immediately.
- Loads MiniLM sentence transformer only during speech search, then unloads it immediately.
- Never runs visual and text embedding models in GPU memory simultaneously.
"""

from typing import Any

from videosdk.config import Config
from videosdk.models.clip_encoder import CLIPEncoder
from videosdk.models.memory import get_device
from videosdk.models.text_encoder import TextEncoder
from videosdk.search.fusion import reciprocal_rank_fusion
from videosdk.store.metadata import MetadataStore
from videosdk.store.vectors import VectorStore
from videosdk.types import SearchResult


def format_timestamp(seconds: float) -> str:
    """Convert a timestamp in seconds to a human-readable MM:SS or H:MM:SS string.

    Args:
        seconds: Non-negative timestamp in seconds.

    Returns:
        Formatted string (e.g., 75.0 -> '01:15', 3661.0 -> '1:01:01').
    """
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class Searcher:
    """Coordinates visual, speech, and hybrid multi-modal video retrieval."""

    def __init__(
        self,
        visual_store: VectorStore,
        text_store: VectorStore,
        metadata_store: MetadataStore,
        config: Config | None = None,
    ) -> None:
        """Initialize Searcher with backing vector and metadata stores.

        Args:
            visual_store: FAISS vector index storing CLIP visual embeddings.
            text_store: FAISS vector index storing MiniLM speech transcript embeddings.
            metadata_store: SQLite relational store for video files and segments.
            config: Optional configuration instance.
        """
        self.visual_store = visual_store
        self.text_store = text_store
        self.metadata_store = metadata_store
        self.config = config or Config()

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "both",
        device: Any = "auto",
    ) -> list[SearchResult]:
        """Perform semantic search across indexed videos.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results to return.
            mode: Search modality:
                - 'visual': Query matches visual appearance using OpenCLIP.
                - 'speech': Query matches spoken transcript using MiniLM.
                - 'both': Evaluates both modalities and merges with RRF.
            device: Compute device ('auto', 'cuda', 'cpu') for encoders.

        Returns:
            List of SearchResult objects sorted by descending relevance.

        Raises:
            ValueError: If an unsupported search mode is specified.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        mode = mode.lower().strip()
        if mode not in ("visual", "speech", "both"):
            raise ValueError(
                f"Unsupported search mode: '{mode}'. Allowed modes: 'visual', 'speech', 'both'."
            )

        if mode == "visual":
            visual_results = self._visual_search(clean_query, top_k=top_k, device=device)
            return self._build_results(visual_results, source="visual")

        if mode == "speech":
            speech_results = self._speech_search(clean_query, top_k=top_k, device=device)
            return self._build_results(speech_results, source="speech")

        # Hybrid 'both' mode: retrieve larger candidate pools for robust fusion
        candidate_k = max(top_k * 2, 20)
        visual_candidates = self._visual_search(clean_query, top_k=candidate_k, device=device)
        speech_candidates = self._speech_search(clean_query, top_k=candidate_k, device=device)

        visual_ids = {seg_id for seg_id, _ in visual_candidates}
        speech_ids = {seg_id for seg_id, _ in speech_candidates}

        # Combine ranked lists with Reciprocal Rank Fusion
        fused_ranked = reciprocal_rank_fusion(
            [visual_candidates, speech_candidates],
            k=self.config.rrf_k,
        )

        results: list[SearchResult] = []
        for seg_id, rrf_score in fused_ranked[:top_k]:
            seg = self.metadata_store.get_segment(seg_id)
            if not seg:
                continue

            # Determine source attribution
            in_visual = seg_id in visual_ids
            in_speech = seg_id in speech_ids
            if in_visual and in_speech:
                source = "both"
            elif in_visual:
                source = "visual"
            else:
                source = "speech"

            start_time = float(seg["start_time"])
            end_time = float(seg["end_time"])

            results.append(
                SearchResult(
                    video=str(seg["file_path"]),
                    start=start_time,
                    end=end_time,
                    timestamp=format_timestamp(start_time),
                    score=round(float(rrf_score), 5),
                    text=str(seg.get("transcript_text", "") or ""),
                    source=source,
                )
            )

        return results

    def _visual_search(
        self,
        query: str,
        top_k: int,
        device: Any,
    ) -> list[tuple[int, float]]:
        """Encode query with CLIP, search visual index, and map to segment IDs."""
        if self.visual_store.count() == 0:
            return []

        resolved_device = get_device(device if isinstance(device, str) else "auto")
        encoder = CLIPEncoder(config=self.config, device=resolved_device)
        encoder.load()
        try:
            query_vec = encoder.encode_text(query)
        finally:
            encoder.unload()

        scores, visual_ids = self.visual_store.search(query_vec, top_k=top_k)

        ranked: list[tuple[int, float]] = []
        for score, v_id in zip(scores, visual_ids, strict=False):
            if v_id < 0:
                continue
            seg = self.metadata_store.get_segment_by_visual_index(int(v_id))
            if seg:
                ranked.append((int(seg["segment_id"]), float(score)))

        return ranked

    def _speech_search(
        self,
        query: str,
        top_k: int,
        device: Any,
    ) -> list[tuple[int, float]]:
        """Encode query with MiniLM, search text index, and map to segment IDs."""
        if self.text_store.count() == 0:
            return []

        encoder = TextEncoder(config=self.config)
        encoder.load()
        try:
            query_vec = encoder.encode([query])
        finally:
            encoder.unload()

        scores, text_ids = self.text_store.search(query_vec, top_k=top_k)

        ranked: list[tuple[int, float]] = []
        for score, t_id in zip(scores, text_ids, strict=False):
            if t_id < 0:
                continue
            seg = self.metadata_store.get_segment_by_text_index(int(t_id))
            if seg:
                ranked.append((int(seg["segment_id"]), float(score)))

        return ranked

    def _build_results(
        self,
        ranked_segments: list[tuple[int, float]],
        source: str,
    ) -> list[SearchResult]:
        """Convert a list of (segment_id, score) into typed SearchResult objects."""
        results: list[SearchResult] = []
        for seg_id, score in ranked_segments:
            seg = self.metadata_store.get_segment(seg_id)
            if not seg:
                continue

            start_time = float(seg["start_time"])
            end_time = float(seg["end_time"])

            results.append(
                SearchResult(
                    video=str(seg["file_path"]),
                    start=start_time,
                    end=end_time,
                    timestamp=format_timestamp(start_time),
                    score=round(float(score), 4),
                    text=str(seg.get("transcript_text", "") or ""),
                    source=source,
                )
            )

        return results
