"""Phase 1 Visual Search Prototype Demo.

Usage:
    python examples/visual_search_demo.py <path_to_video.mp4> "<search_query>"

Pipeline:
1. Scene Detection: Splits video into semantically coherent segments using OpenCV HSV histograms.
2. Visual Embedding: Extracts representative frames per segment, embeds with OpenCLIP, averages.
3. Indexing: Stores segment vectors in a local in-memory FAISS IndexFlatIP.
4. Search: Embeds text query with OpenCLIP text encoder, queries FAISS for top-5 segments.
5. Unloads models to respect 4 GB VRAM limit.
"""

import os
import sys
from pathlib import Path

# Prefer local offline cache to prevent network timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from videosdk.config import Config
from videosdk.models.clip_encoder import CLIPEncoder, embed_segments
from videosdk.models.memory import get_device, get_gpu_memory
from videosdk.store.vectors import VectorStore
from videosdk.video.scenes import detect_scenes


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or H:MM:SS string."""
    total_seconds = round(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_visual_search(video_path: str, query: str, top_k: int = 5) -> None:
    path = Path(video_path)
    if not path.is_file():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    config = Config()
    device = get_device("auto")

    print("\n--- VideoSDK Phase 1 Visual Search Prototype ---")
    print(f"Target video : {path.name}")
    print(f"Search query : '{query}'")
    print(f"Compute device: {device}")
    mem = get_gpu_memory()
    if mem["total_mb"] > 0:
        print(f"GPU memory   : {mem['free_mb']:.1f} MB free / {mem['total_mb']:.1f} MB total")

    # Step 1: Detect scenes
    print("\n[1/4] Detecting scene boundaries...")
    segments = detect_scenes(path, config)
    print(f"      Detected {len(segments)} segments.")

    # Step 2: Embed segments with CLIP
    print("\n[2/4] Embedding visual content (OpenCLIP)...")
    segment_embeddings = embed_segments(path, segments, config, device)
    print(f"      Generated embedding matrix: {segment_embeddings.shape}")

    # Step 3: Populate FAISS index
    print("\n[3/4] Populating FAISS vector index...")
    index_path = path.parent / f"{path.stem}_visual.faiss"
    store = VectorStore(path=index_path, dim=config.clip_dim)
    store.create()
    store.add(segment_embeddings)
    print(f"      Indexed {store.count()} segment vectors.")

    # Step 4: Encode text query and search
    print(f"\n[4/4] Encoding query and searching top-{top_k} results...")
    encoder = CLIPEncoder(config=config, device=device)
    encoder.load()
    try:
        query_vec = encoder.encode_text(query)
    finally:
        encoder.unload()

    scores, indices = store.search(query_vec, top_k=top_k)

    print("\n--- Top Search Results ---")
    for rank, (score, seg_idx) in enumerate(zip(scores, indices, strict=False), start=1):
        start_sec, end_sec = segments[seg_idx]
        ts_str = format_timestamp(start_sec)
        time_range = f"{start_sec:.1f}s - {end_sec:.1f}s"
        print(f"#{rank} | Score: {score:0.4f} | Timestamp: {ts_str} ({time_range})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python examples/visual_search_demo.py <video_path> <query>")
        sys.exit(1)

    video_input = sys.argv[1]
    query_input = sys.argv[2]
    run_visual_search(video_input, query_input)
