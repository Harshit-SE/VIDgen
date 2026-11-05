"""Central configuration — all magic numbers and model names live here."""

from dataclasses import dataclass


@dataclass
class Config:
    # Scene detection
    scene_hist_threshold: float = 0.7
    scene_sample_interval: float = 0.5  # seconds between sampled frames
    segment_min_seconds: float = 2.0
    segment_max_seconds: float = 30.0
    segment_fallback_seconds: float = 10.0

    # CLIP
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_dim: int = 512
    frames_per_segment: int = 3
    clip_batch_size: int = 32

    # Whisper
    whisper_model_size: str = "base"
    whisper_compute_type_gpu: str = "int8"
    whisper_compute_type_cpu: str = "float32"  # int8 not supported on all CPUs
    whisper_vad_filter: bool = True
    whisper_min_confidence: float = -1.0  # avg_logprob threshold

    # Text Embeddings
    text_model_name: str = "all-MiniLM-L6-v2"
    text_dim: int = 384
    text_batch_size: int = 64

    # RRF
    rrf_k: int = 60

    # Index
    schema_version: int = 1

    # Memory
    min_batch_size: int = 1
    oom_batch_shrink_factor: int = 2
