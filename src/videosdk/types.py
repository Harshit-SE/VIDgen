"""Typed data objects returned by the public API."""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single search result pointing to a moment in a video."""

    video: str  # path of the source video
    start: float  # start time in seconds
    end: float  # end time in seconds
    timestamp: str  # human-readable "MM:SS" or "H:MM:SS"
    score: float  # higher is better
    text: str  # matched transcript text (may be empty)
    source: str  # "visual" | "speech" | "both"

    def clip(self, output_path: str) -> str:
        """Save a short clip of this result to output_path. Returns the path."""
        from videosdk.video.clipper import cut_clip

        return cut_clip(self.video, self.start, self.end, output_path)
