"""OpenCLIP visual and text encoder wrapper for video segment retrieval.

Enforces memory safety for 4GB VRAM environments:
- Lazy loading: weights loaded only during active inference
- Half-precision (fp16) on CUDA devices
- L2 normalization for all vectors (cosine similarity = dot product)
- Explicit model unloading and cache evacuation after segment processing
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from videosdk.config import Config
from videosdk.errors import VideoNotFoundError
from videosdk.models.memory import get_device, unload_model
from videosdk.video.reader import iter_frames_at_times

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """Wrapper around OpenCLIP for embedding image frames and search queries."""

    def __init__(
        self,
        config: Config | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or Config()
        self.device = device or get_device("auto")
        self.model: Any = None
        self.preprocess: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        """Load OpenCLIP weights and tokenizer to the configured device."""
        if self.model is not None:
            return

        import os

        import open_clip

        # When offline or cached, skip remote check to avoid HTTP timeout delays
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        logger.info(
            "Loading CLIP model %s (%s) on %s",
            self.config.clip_model_name,
            self.config.clip_pretrained,
            self.device,
        )

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.clip_model_name,
                pretrained=self.config.clip_pretrained,
                device=self.device,
            )
        except Exception as e:
            # If network error or timeout occurs, attempt loading from offline local cache
            logger.warning("Could not contact remote hub (%s). Trying offline cache.", e)
            os.environ["HF_HUB_OFFLINE"] = "1"
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.clip_model_name,
                pretrained=self.config.clip_pretrained,
                device=self.device,
            )
        model.eval()

        if self.device.type == "cuda":
            model.to(torch.float16)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)

    def encode_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Encode a sequence of RGB image frames into L2-normalized 512-dim vectors.

        Args:
            frames: Sequence of uint8 RGB numpy arrays of shape (H, W, 3).

        Returns:
            2D numpy array of shape (len(frames), clip_dim) with dtype float32.
        """
        if not frames:
            return np.empty((0, self.config.clip_dim), dtype=np.float32)

        if self.model is None or self.preprocess is None:
            self.load()

        processed_images = []
        for frame in frames:
            pil_img = Image.fromarray(frame)
            processed_images.append(self.preprocess(pil_img))

        image_tensor = torch.stack(processed_images).to(self.device)
        if self.device.type == "cuda":
            image_tensor = image_tensor.to(torch.float16)

        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            # L2-normalize vectors
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().to(torch.float32).numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query string into an L2-normalized 1x512 vector.

        Args:
            text: Query string.

        Returns:
            2D numpy array of shape (1, clip_dim) with dtype float32.
        """
        if self.model is None or self.tokenizer is None:
            self.load()

        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().to(torch.float32).numpy()

    def unload(self) -> None:
        """Explicitly purge model weights and free GPU memory."""
        unload_model(self.model)
        self.model = None
        self.preprocess = None
        self.tokenizer = None


def embed_segments(
    path: str | Path,
    segments: Sequence[tuple[float, float]],
    config: Config | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Extract and embed representative frames from segments, then unload CLIP.

    For each segment, extracts `frames_per_segment` (default 3) evenly spaced frames,
    encodes them in batches, averages the frame vectors for each segment, and
    re-normalizes the resulting segment vector.

    Args:
        path: Path to the video file.
        segments: Sequence of (start_sec, end_sec) segment boundaries.
        config: Optional configuration.
        device: Device to use for CLIP.

    Returns:
        NumPy matrix of shape (num_segments, clip_dim) with float32 values.

    Raises:
        VideoNotFoundError: If video does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise VideoNotFoundError(str(path))

    cfg = config or Config()
    if not segments:
        return np.empty((0, cfg.clip_dim), dtype=np.float32)

    # 1. Determine sample timestamps for each segment
    all_timestamps: list[float] = []
    segment_frame_counts: list[int] = []

    for start, end in segments:
        dur = max(0.01, end - start)
        k = cfg.frames_per_segment
        if k == 1:
            offsets = [dur * 0.5]
        elif k == 2:
            offsets = [dur * 0.25, dur * 0.75]
        else:
            # 3 or more frames
            offsets = [dur * (i + 1) / (k + 1) for i in range(k)]

        seg_times = [round(start + off, 3) for off in offsets]
        all_timestamps.extend(seg_times)
        segment_frame_counts.append(len(seg_times))

    # 2. Extract frames via generator
    frames = list(iter_frames_at_times(file_path, all_timestamps))

    # 3. Load encoder and compute embeddings in batches
    encoder = CLIPEncoder(config=cfg, device=device)
    encoder.load()

    all_embeddings: list[np.ndarray] = []
    batch_size = cfg.clip_batch_size

    try:
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            emb = encoder.encode_frames(batch)
            all_embeddings.append(emb)

        frame_matrix = (
            np.vstack(all_embeddings)
            if all_embeddings
            else np.empty((0, cfg.clip_dim), dtype=np.float32)
        )

        # 4. Average frame vectors per segment and L2 normalize
        segment_embeddings = np.zeros((len(segments), cfg.clip_dim), dtype=np.float32)
        idx = 0
        for s_idx, count in enumerate(segment_frame_counts):
            seg_frames = frame_matrix[idx : idx + count]
            idx += count

            if len(seg_frames) > 0:
                mean_vec = np.mean(seg_frames, axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-6:
                    mean_vec = mean_vec / norm
                segment_embeddings[s_idx] = mean_vec

        return segment_embeddings

    finally:
        # Crucial: always unload model to stay under 4 GB VRAM limit
        encoder.unload()
