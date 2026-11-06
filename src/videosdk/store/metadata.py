"""SQLite metadata store for video files and segments.

Persists structured video metadata, segment timestamps, transcript text,
and mappings to visual and text vector store row positions.
"""

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MetadataStore:
    """Manages relational metadata for indexed videos and segments via SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the metadata store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._connect()

    def _connect(self) -> None:
        """Establish SQLite connection and configure pragmas."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        # Enforce foreign key constraints (cascading deletes)
        self._conn.execute("PRAGMA foreign_keys = ON;")

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the active SQLite connection."""
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        return self._conn

    def create_tables(self) -> None:
        """Create the videos and segments tables if they do not already exist."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path   TEXT NOT NULL UNIQUE,
                    file_hash   TEXT NOT NULL,
                    duration    REAL NOT NULL,
                    fps         REAL NOT NULL,
                    width       INTEGER NOT NULL,
                    height      INTEGER NOT NULL,
                    added_at    TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id        INTEGER NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
                    start_time      REAL NOT NULL,
                    end_time        REAL NOT NULL,
                    transcript_text TEXT DEFAULT '',
                    visual_index_id INTEGER NOT NULL,
                    text_index_id   INTEGER DEFAULT -1,
                    UNIQUE(video_id, start_time)
                );
                """
            )
            # Create indexes for fast lookup during search fusion
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_visual ON segments(visual_index_id);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_text ON segments(text_index_id);"
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_hash ON videos(file_hash);")

    def add_video(
        self,
        file_path: str,
        file_hash: str,
        duration: float,
        fps: float,
        width: int,
        height: int,
        added_at: str | None = None,
    ) -> int:
        """Insert a video record into the videos table.

        Args:
            file_path: Path of the source video.
            file_hash: SHA-256 hash of the video file.
            duration: Total duration in seconds.
            fps: Frame rate.
            width: Video frame width in pixels.
            height: Video frame height in pixels.
            added_at: ISO-8601 timestamp string (defaults to current UTC time).

        Returns:
            The newly created video_id (integer primary key).
        """
        if added_at is None:
            added_at = datetime.now(timezone.utc).isoformat()

        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO videos (file_path, file_hash, duration, fps, width, height, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(file_path),
                    file_hash,
                    float(duration),
                    float(fps),
                    int(width),
                    int(height),
                    added_at,
                ),
            )
            return int(cursor.lastrowid)

    def add_segments(
        self,
        video_id: int,
        segments_data: Sequence[dict[str, Any]],
    ) -> None:
        """Bulk insert segments for an indexed video.

        Args:
            video_id: Foreign key referencing videos(video_id).
            segments_data: Sequence of dictionaries containing:
                - start_time (float)
                - end_time (float)
                - transcript_text (str, optional)
                - visual_index_id (int)
                - text_index_id (int, optional, default -1)
        """
        if not segments_data:
            return

        rows = []
        for s in segments_data:
            rows.append(
                (
                    video_id,
                    float(s["start_time"]),
                    float(s["end_time"]),
                    str(s.get("transcript_text", "")),
                    int(s["visual_index_id"]),
                    int(s.get("text_index_id", -1)),
                )
            )

        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO segments (
                    video_id, start_time, end_time, transcript_text, visual_index_id, text_index_id
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                rows,
            )

    def get_video(self, file_path: str) -> dict[str, Any] | None:
        """Find video by its file path."""
        cursor = self.conn.execute(
            "SELECT * FROM videos WHERE file_path = ?;",
            (str(file_path),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_video_by_id(self, video_id: int) -> dict[str, Any] | None:
        """Find video by its primary key ID."""
        cursor = self.conn.execute(
            "SELECT * FROM videos WHERE video_id = ?;",
            (int(video_id),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_video_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """Find video by its SHA-256 file hash."""
        cursor = self.conn.execute(
            "SELECT * FROM videos WHERE file_hash = ?;",
            (file_hash,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def has_video_hash(self, file_hash: str) -> bool:
        """Check if a video with the given SHA-256 hash has already been indexed."""
        cursor = self.conn.execute(
            "SELECT 1 FROM videos WHERE file_hash = ? LIMIT 1;",
            (file_hash,),
        )
        return cursor.fetchone() is not None

    def get_segment(self, segment_id: int) -> dict[str, Any] | None:
        """Retrieve a segment record by its primary key."""
        cursor = self.conn.execute(
            """
            SELECT s.*, v.file_path
            FROM segments s
            JOIN videos v ON s.video_id = v.video_id
            WHERE s.segment_id = ?;
            """,
            (int(segment_id),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_segments_for_video(self, video_id: int) -> list[dict[str, Any]]:
        """Retrieve all segments belonging to a given video ID, ordered by start_time."""
        cursor = self.conn.execute(
            """
            SELECT s.*, v.file_path
            FROM segments s
            JOIN videos v ON s.video_id = v.video_id
            WHERE s.video_id = ?
            ORDER BY s.start_time ASC;
            """,
            (int(video_id),),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_segment_by_visual_index(self, visual_index_id: int) -> dict[str, Any] | None:
        """Look up a segment by its row position in visual.faiss."""
        cursor = self.conn.execute(
            """
            SELECT s.*, v.file_path
            FROM segments s
            JOIN videos v ON s.video_id = v.video_id
            WHERE s.visual_index_id = ?;
            """,
            (int(visual_index_id),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_segment_by_text_index(self, text_index_id: int) -> dict[str, Any] | None:
        """Look up a segment by its row position in text.faiss."""
        cursor = self.conn.execute(
            """
            SELECT s.*, v.file_path
            FROM segments s
            JOIN videos v ON s.video_id = v.video_id
            WHERE s.text_index_id = ?;
            """,
            (int(text_index_id),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_videos(self) -> list[dict[str, Any]]:
        """List all indexed video records with segment counts."""
        cursor = self.conn.execute(
            """
            SELECT v.*, COUNT(s.segment_id) as segment_count
            FROM videos v
            LEFT JOIN segments s ON v.video_id = s.video_id
            GROUP BY v.video_id
            ORDER BY v.added_at DESC;
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def remove_video(self, file_path: str) -> bool:
        """Remove a video by file path.

        Cascading delete will automatically delete associated segments.

        Returns:
            True if a video was removed, False if no matching video was found.
        """
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM videos WHERE file_path = ?;",
                (str(file_path),),
            )
            return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "MetadataStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
