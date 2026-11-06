"""Automatic Speech Recognition (ASR) wrapper using faster-whisper.

Transcribes audio using CTranslate2-accelerated Whisper models with Silero Voice Activity
Detection (VAD) to skip silences and prevent hallucinated text. Enforces sequential model
unloading to respect the 4 GB VRAM limit.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from videosdk.config import Config
from videosdk.models.memory import get_device, unload_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptChunk:
    """A segment of transcribed speech with temporal bounds and confidence."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str  # Decoded text content
    confidence: float  # Average log-probability score (higher / closer to 0 is better)


class WhisperTranscriber:
    """Faster-Whisper model lifecycle wrapper with VAD filtering and memory reclamation."""

    def __init__(
        self,
        config: Config | None = None,
        device: Any = "auto",
    ) -> None:
        """Initialize the transcriber. Does not load the neural model into memory yet.

        Args:
            config: VideoSDK configuration instance.
            device: Requested device ('auto', 'cuda', 'cpu', or torch.device).
        """
        self.config = config or Config()
        self.device_pref = device
        self.model: Any = None
        self._resolved_device: str = "cpu"

    def load(self) -> None:
        """Instantiate and load the faster-whisper model into memory."""
        if self.model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as err:
            raise ImportError(
                "faster-whisper is required for speech transcription. "
                "Install it with: pip install faster-whisper"
            ) from err

        # Resolve compute device
        if hasattr(self.device_pref, "type"):
            device_str = str(self.device_pref.type)
        elif isinstance(self.device_pref, str):
            resolved = get_device(self.device_pref)
            device_str = str(resolved.type) if hasattr(resolved, "type") else str(resolved)
        else:
            device_str = "cpu"

        self._resolved_device = device_str

        # Select quantization compute type based on device
        if device_str == "cuda":
            compute_type = self.config.whisper_compute_type_gpu
        else:
            compute_type = self.config.whisper_compute_type_cpu

        logger.info(
            "Loading faster-whisper '%s' on %s (compute_type=%s)...",
            self.config.whisper_model_size,
            device_str,
            compute_type,
        )

        try:
            self.model = WhisperModel(
                self.config.whisper_model_size,
                device=device_str,
                compute_type=compute_type,
            )
        except Exception as e:
            # Fallback to CPU float32 if GPU initialization or quantization fails
            if device_str == "cuda":
                logger.warning("Failed to initialize Whisper on CUDA (%s). Falling back to CPU.", e)
                self._resolved_device = "cpu"
                self.model = WhisperModel(
                    self.config.whisper_model_size,
                    device="cpu",
                    compute_type="float32",
                )
            else:
                raise

    def transcribe(self, audio_path: str | Path) -> list[TranscriptChunk]:
        """Transcribe an audio file with voice activity detection and confidence filtering.

        Args:
            audio_path: Path to the 16 kHz mono WAV file.

        Returns:
            List of filtered TranscriptChunk objects.
        """
        if self.model is None:
            self.load()

        src = Path(audio_path)
        if not src.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        vad_params = dict(min_silence_duration_ms=500) if self.config.whisper_vad_filter else None

        segments_gen, _info = self.model.transcribe(
            str(src),
            vad_filter=self.config.whisper_vad_filter,
            vad_parameters=vad_params,
        )

        results: list[TranscriptChunk] = []
        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue

            # Drop very low confidence segments to eliminate noise / hallucinations
            avg_logprob = getattr(seg, "avg_logprob", 0.0)
            if avg_logprob < self.config.whisper_min_confidence:
                logger.debug(
                    "Dropping low-confidence transcript (avg_logprob=%.2f < threshold=%.2f): '%s'",
                    avg_logprob,
                    self.config.whisper_min_confidence,
                    text,
                )
                continue

            results.append(
                TranscriptChunk(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                    confidence=float(avg_logprob),
                )
            )

        return results

    def unload(self) -> None:
        """Purge model from memory and release VRAM/RAM immediately."""
        if self.model is not None:
            unload_model(self.model)
            self.model = None


def align_transcript_to_segments(
    chunks: Sequence[TranscriptChunk],
    segments: Sequence[tuple[float, float]],
) -> dict[int, str]:
    """Align timestamped transcript chunks to video segments based on temporal overlap.

    A transcript chunk overlaps a segment [seg_start, seg_end] if:
        chunk.start < seg_end and chunk.end > seg_start

    Args:
        chunks: Sequence of transcribed chunks with start/end timestamps.
        segments: Sequence of (start_seconds, end_seconds) segment tuples.

    Returns:
        Mapping of segment index to the combined transcript text spoken within it.
    """
    aligned: dict[int, str] = {}

    for idx, (seg_start, seg_end) in enumerate(segments):
        matching_texts: list[str] = []
        for chunk in chunks:
            # Check for temporal overlap
            if chunk.start < seg_end and chunk.end > seg_start:
                matching_texts.append(chunk.text)

        aligned[idx] = " ".join(matching_texts).strip()

    return aligned
