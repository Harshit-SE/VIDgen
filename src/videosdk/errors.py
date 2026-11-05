"""Custom exceptions with actionable error messages."""


class VideoSDKError(Exception):
    """Base exception for all VideoSDK errors."""


class VideoNotFoundError(VideoSDKError):
    """Raised when a video file does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Video file not found: {path}")


class FFmpegNotFoundError(VideoSDKError):
    """Raised when FFmpeg is not installed or not on PATH."""

    def __init__(self) -> None:
        super().__init__(
            "FFmpeg not found. Install it:\n"
            "  Windows: winget install Gyan.FFmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )


class IndexVersionError(VideoSDKError):
    """Raised when the on-disk index version doesn't match."""

    def __init__(self, found: int, expected: int) -> None:
        super().__init__(
            f"Index version mismatch: found v{found}, expected v{expected}. "
            "Re-index your videos with the current version."
        )


class InvalidVideoError(VideoSDKError):
    """Raised when a file cannot be read as a video."""

    def __init__(self, path: str, reason: str = "") -> None:
        msg = f"Cannot read as video: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)
