"""Unit tests for Whisper ASR model wrapper and transcript alignment in videosdk.models.asr."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from videosdk.config import Config
from videosdk.models.asr import (
    TranscriptChunk,
    WhisperTranscriber,
    align_transcript_to_segments,
)


def test_transcript_chunk_creation() -> None:
    """Test creating a TranscriptChunk dataclass."""
    chunk = TranscriptChunk(start=1.5, end=4.2, text="hello world", confidence=-0.35)
    assert chunk.start == 1.5
    assert chunk.end == 4.2
    assert chunk.text == "hello world"
    assert chunk.confidence == -0.35


def test_align_transcript_to_segments() -> None:
    """Test temporal alignment of speech chunks to video scene segments."""
    chunks = [
        TranscriptChunk(start=1.0, end=3.0, text="Introduction.", confidence=-0.2),
        TranscriptChunk(start=3.5, end=5.0, text="Deep learning.", confidence=-0.1),
        TranscriptChunk(start=7.0, end=9.5, text="Backpropagation.", confidence=-0.3),
        TranscriptChunk(start=12.0, end=14.0, text="Conclusion.", confidence=-0.1),
    ]

    segments = [
        (0.0, 6.0),  # Overlaps chunks 0 and 1
        (6.0, 10.0),  # Overlaps chunk 2
        (10.0, 11.5),  # Silent gap (no chunks)
        (11.5, 15.0),  # Overlaps chunk 3
    ]

    aligned = align_transcript_to_segments(chunks, segments)
    assert len(aligned) == 4
    assert aligned[0] == "Introduction. Deep learning."
    assert aligned[1] == "Backpropagation."
    assert aligned[2] == ""  # No speech in this segment
    assert aligned[3] == "Conclusion."


def test_align_transcript_boundary_overlap() -> None:
    """Test that boundary edge cases (touching timestamps) do not erroneously overlap."""
    chunks = [
        TranscriptChunk(start=0.0, end=5.0, text="Part 1", confidence=0.0),
        TranscriptChunk(start=5.0, end=10.0, text="Part 2", confidence=0.0),
    ]
    segments = [
        (0.0, 5.0),
        (5.0, 10.0),
    ]
    aligned = align_transcript_to_segments(chunks, segments)
    assert aligned[0] == "Part 1"
    assert aligned[1] == "Part 2"


def test_whisper_transcriber_load_missing_faster_whisper() -> None:
    """Test that missing faster_whisper package raises an actionable ImportError."""
    transcriber = WhisperTranscriber()
    with (
        patch.dict(sys.modules, {"faster_whisper": None}),
        pytest.raises(ImportError, match="faster-whisper is required"),
    ):
        transcriber.load()


def test_whisper_transcriber_mocked_transcription(tmp_path: Path) -> None:
    """Test transcription with VAD filtering and confidence thresholding."""
    dummy_audio = tmp_path / "speech.wav"
    dummy_audio.write_bytes(b"RIFF dummy wav data")

    # Mock whisper segment objects
    seg1 = MagicMock(start=0.0, end=2.5, text=" Valid speech. ", avg_logprob=-0.4)
    # seg2 has avg_logprob < -1.0 threshold -> should be dropped
    seg2 = MagicMock(start=2.6, end=4.0, text=" Hallucinated noise ", avg_logprob=-1.5)
    seg3 = MagicMock(start=4.5, end=6.0, text="   ", avg_logprob=-0.2)  # whitespace only
    seg4 = MagicMock(start=6.1, end=8.0, text=" Another sentence. ", avg_logprob=-0.3)

    mock_whisper_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.transcribe.return_value = ([seg1, seg2, seg3, seg4], None)
    mock_whisper_model_cls.return_value = mock_instance

    mock_faster_whisper = ModuleType("faster_whisper")
    mock_faster_whisper.WhisperModel = mock_whisper_model_cls  # type: ignore[attr-defined]

    config = Config(whisper_min_confidence=-1.0)
    transcriber = WhisperTranscriber(config=config, device="cpu")

    with patch.dict(sys.modules, {"faster_whisper": mock_faster_whisper}):
        transcriber.load()
        results = transcriber.transcribe(dummy_audio)

        assert len(results) == 2
        assert results[0].text == "Valid speech."
        assert results[0].confidence == -0.4
        assert results[1].text == "Another sentence."
        assert results[1].confidence == -0.3

        # Test unloading
        transcriber.unload()
        assert transcriber.model is None


def test_whisper_transcriber_missing_audio_file() -> None:
    """Test that transcribe() raises FileNotFoundError when audio file is missing."""
    transcriber = WhisperTranscriber()
    transcriber.model = MagicMock()
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe("nonexistent_audio.wav")
