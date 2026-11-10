# VideoSDK — Complete Implementation Guide

> **Purpose:** This is the single source of truth for *how* to build every module.
> At each phase, open this file, follow the instructions exactly, then tick off the phase in `TRACKER.md` and document detailed phase notes, architectural decisions, and key code walkthroughs in `notes.md`.
> Do not deviate from the algorithms, libraries, or data flows described here without updating this document first.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Tech Stack — What We Use and Why](#2-tech-stack--what-we-use-and-why)
3. [Core Algorithms Explained](#3-core-algorithms-explained)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [On-Disk Index Format](#5-on-disk-index-format)
6. [Memory Management Strategy](#6-memory-management-strategy)
7. [Phase 0 — Setup and Decisions](#phase-0--setup-and-decisions)
8. [Phase 1 — Visual Search Prototype](#phase-1--visual-search-prototype)
9. [Phase 2 — Speech Search](#phase-2--speech-search)
10. [Phase 3 — Combine and Rank (Fusion)](#phase-3--combine-and-rank-fusion)
11. [Phase 4 — Turn It Into an SDK](#phase-4--turn-it-into-an-sdk)
12. [Phase 4.5 — Advanced Multimodal Intelligence & Creator Tools](#phase-45--advanced-multimodal-intelligence--creator-tools)
13. [Phase 5 — Tests, Eval Set, and Benchmarks](#phase-5--tests-eval-set-and-benchmarks)
14. [Phase 6 — Docs, Polish, and Release](#phase-6--docs-polish-and-release)
15. [Error Handling Strategy](#15-error-handling-strategy)
16. [Testing Strategy](#16-testing-strategy)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                        │
│                  Python API (VideoIndex)  +  CLI (Typer)           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                        Orchestration Layer                          │
│                     index.py  (VideoIndex class)                    │
│         add() → indexing pipeline    search() → search pipeline     │
└──────┬────────────────┬───────────────┬────────────────┬────────────┘
       │                │               │                │
┌──────▼──────┐  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│   video/    │  │   models/   │ │   store/    │ │   search/   │
│ reader.py   │  │clip_encoder │ │ vectors.py  │ │ fusion.py   │
│ scenes.py   │  │  asr.py     │ │ metadata.py │ │ searcher.py │
│ clipper.py  │  │text_encoder │ │             │ │             │
│             │  │  memory.py  │ │             │ │             │
└─────────────┘  └─────────────┘ └─────────────┘ └─────────────┘
```

**Key architectural rule:** Models are loaded ONE AT A TIME, never simultaneously.
After each model finishes its work, it is fully unloaded and GPU memory is freed
before the next model loads. This is how we fit inside 4 GB VRAM.

---

## 2. Tech Stack — What We Use and Why

### 2.1 Video Reading & Scene Detection — OpenCV (`opencv-python-headless`)

**What it does:** Reads video frames, extracts metadata (fps, resolution, duration, frame count), and detects scene boundaries using histogram comparison.

**Why OpenCV:**
- Industry standard for video frame operations.
- `headless` variant has no GUI dependencies — perfect for servers and CI.
- Built-in histogram comparison functions (`cv2.compareHist`) are fast and reliable for scene detection without needing a neural network.
- Reads frames lazily (one at a time) so we never load the entire video into RAM.

**Alternatives rejected:**
- `decord`: Faster for batch frame reading but less mature, fewer platform builds, harder to install on Windows.
- `imageio` / `moviepy`: Higher-level wrappers that add unnecessary abstraction and still depend on FFmpeg internally.
- `PyAV`: Powerful but complex API for our simple needs.

### 2.2 Audio Extraction & Video Clipping — FFmpeg (system binary)

**What it does:** Extracts the audio track from a video file (to WAV for Whisper), and cuts short clips from a video by timestamp.

**Why FFmpeg:**
- The universal standard for media manipulation. Nothing else comes close in reliability.
- Invoked as a subprocess — no Python binding needed, no memory footprint.
- Stream-copy clipping (`-c copy`) is nearly instant (no re-encoding).
- Already required by `faster-whisper` internally.

**How we call it:** Via Python's `subprocess.run()` with explicit argument lists (never shell=True). We validate that `ffmpeg` is on PATH at startup and give a clear error if missing.

### 2.3 Visual Embeddings — CLIP ViT-B/32 via `open_clip_torch`

**What it does:** Converts images (video frames) and text queries into 512-dimensional vectors in a shared embedding space, so we can measure similarity between "what a frame shows" and "what the user typed."

**Why CLIP ViT-B/32:**
- Trained by OpenAI on 400M image-text pairs — understands a huge range of visual concepts.
- ViT-B/32 is the smallest CLIP variant: ~150 MB weights, ~600 MB GPU memory in fp16.
- Fits comfortably in 4 GB VRAM alongside batch processing overhead.
- Text and image embeddings live in the same vector space — no separate training needed.

**Why `open_clip_torch`:**
- Clean, well-maintained reimplementation of CLIP with many model variants.
- Simple API: `create_model_and_transforms()` → model + preprocessor.
- Supports fp16 inference natively.
- Active development, good community.

**Alternatives rejected:**
- `transformers` (HuggingFace): Works but heavier import, slower model loading, more abstraction layers.
- Original `clip` package (OpenAI): Archived, no longer maintained.
- Larger CLIP models (ViT-L/14, ViT-H): Better accuracy but 2-4x more VRAM — won't fit in 4 GB.

**The embedding process:**
1. Load the CLIP model + image preprocessor + tokenizer.
2. For each video segment, extract representative frames (start, middle, end of segment).
3. Preprocess frames (resize to 224×224, normalize).
4. Run `model.encode_image(batch)` → get 512-dim vectors.
5. Average the frame vectors for each segment → one vector per segment.
6. Unload model, free GPU.

**For search:**
1. Load only the CLIP text encoder (or the full model — text encoding is lightweight).
2. Tokenize the query string.
3. Run `model.encode_text(tokens)` → one 512-dim query vector.
4. Compare against stored visual vectors using FAISS.

### 2.4 Speech-to-Text — `faster-whisper` (CTranslate2 backend)

**What it does:** Transcribes speech in the audio track to text with word-level timestamps.

**Why `faster-whisper`:**
- Up to 4× faster than OpenAI's original Whisper implementation.
- Uses CTranslate2 under the hood — optimized C++ inference with int8/fp16 quantization.
- `base` model: ~150 MB, ~500 MB GPU memory in int8. Fits in 4 GB.
- Built-in Voice Activity Detection (VAD) via Silero — skips silent sections automatically, preventing hallucinated text.
- Word-level timestamps for precise alignment to video segments.

**Alternatives rejected:**
- `openai-whisper`: Original implementation. 2-4× slower, higher memory usage, no built-in VAD.
- `whisperx`: Good but adds complexity (diarization, alignment) we don't need in v1.
- Cloud APIs (Google STT, AWS Transcribe): Violates the "fully offline" requirement.

**The transcription process:**
1. Extract audio from video as 16 kHz mono WAV using FFmpeg.
2. Load `faster-whisper` model (`base` size, int8 quantization on GPU, fp32 on CPU).
3. Run transcription with `vad_filter=True` to skip silence.
4. Collect segments: each has `start`, `end`, `text`, and `avg_logprob` (confidence).
5. Drop segments where `avg_logprob < -1.0` (very low confidence → likely hallucinated).
6. Unload model, free GPU.

### 2.5 Transcript Embeddings — `sentence-transformers` with `all-MiniLM-L6-v2`

**What it does:** Converts transcript text chunks into 384-dimensional vectors for semantic text search.

**Why `all-MiniLM-L6-v2`:**
- Tiny model: ~80 MB weights, runs comfortably on CPU (no GPU needed).
- 384-dimensional output — smaller vectors mean smaller FAISS index on disk.
- Trained specifically for semantic similarity tasks.
- One of the most battle-tested sentence embedding models.

**Why `sentence-transformers`:**
- High-level API: `model.encode(sentences)` — one line.
- Handles tokenization, batching, and pooling internally.
- We run this on CPU deliberately (model is small enough), so GPU is free.

**Alternatives rejected:**
- Larger models (`all-mpnet-base-v2`): Better quality but 3× larger, not worth it for our use case.
- OpenAI embeddings API: Violates offline requirement.
- Using CLIP text encoder for transcript search: CLIP's text encoder is optimized for image-text matching, not text-text semantic similarity. MiniLM is purpose-built for this.

### 2.6 Vector Search — FAISS (`faiss-cpu`)

**What it does:** Stores embedding vectors and finds the most similar ones to a query vector using approximate nearest neighbor search.

**Why FAISS:**
- Built by Meta AI Research. The standard for vector similarity search.
- `IndexFlatIP` (inner product / cosine similarity) is exact — no approximation error for our scale.
- Blazing fast: searching 100K vectors takes <1 ms.
- Simple serialize/deserialize: `faiss.write_index()` / `faiss.read_index()`.
- CPU-only version (`faiss-cpu`) uses zero GPU memory.

**Why `faiss-cpu` not `faiss-gpu`:**
- Our vector counts are small (a 1-hour video ≈ 200-400 segments). CPU search is already sub-millisecond.
- Using `faiss-cpu` means zero GPU memory used during search — all VRAM is free for model loading.

**Alternatives rejected:**
- `chromadb`, `qdrant`, `weaviate`: Full vector databases with servers. Massive overkill for our use case (we're an SDK, not a service).
- `annoy` (Spotify): Good but read-only after build — can't add vectors incrementally.
- `hnswlib`: Fast but FAISS is more widely known and has better serialization.

**Index structure:**
- We maintain TWO separate FAISS indexes:
  - `visual.faiss`: 512-dim vectors from CLIP (one per video segment).
  - `text.faiss`: 384-dim vectors from MiniLM (one per transcript-aligned segment).
- Both use `IndexFlatIP` (exact inner product search) with L2-normalized vectors (equivalent to cosine similarity).

### 2.7 Metadata Storage — SQLite (`sqlite3`)

**What it does:** Stores structured metadata — which videos are indexed, segment boundaries, transcript text, file hashes for deduplication.

**Why SQLite:**
- Built into Python (no install needed).
- Single file on disk — easy to back up, move, delete.
- ACID-compliant — no corruption from crashes.
- Fast enough for our scale (thousands of rows, not millions).

**Schema (two tables):**

```sql
CREATE TABLE videos (
    video_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL UNIQUE,
    file_hash   TEXT NOT NULL,       -- SHA-256 for dedup
    duration    REAL NOT NULL,       -- seconds
    fps         REAL NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    added_at    TEXT NOT NULL        -- ISO-8601 timestamp
);

CREATE TABLE segments (
    segment_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        INTEGER NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    start_time      REAL NOT NULL,   -- seconds
    end_time        REAL NOT NULL,   -- seconds
    transcript_text TEXT DEFAULT '',  -- may be empty if no speech
    visual_index_id INTEGER NOT NULL, -- row position in visual.faiss
    text_index_id   INTEGER DEFAULT -1, -- row position in text.faiss, -1 if no text
    UNIQUE(video_id, start_time)
);
```

### 2.8 CLI — Typer

**Why Typer:** Automatic `--help`, type validation, colored output, minimal code. Built on Click but with type hints.

### 2.9 Packaging — `pyproject.toml` with `hatchling`

**Why hatchling:** Modern, fast, follows PEP 621 exactly. No `setup.py` needed. Clean editable installs.

### 2.10 Code Quality — Ruff + Mypy

**Why Ruff:** Replaces flake8, isort, black in one tool. Extremely fast (written in Rust).
**Why Mypy:** Catches type errors before runtime. We use `--strict` on public API modules.

---

## 3. Core Algorithms Explained

### 3.1 Scene Detection Algorithm (OpenCV Histogram Comparison)

**Goal:** Split a video into semantically meaningful segments at shot boundaries.

**Algorithm:**

```
1. Read frames at a sampling rate (every 0.5 seconds, not every frame — saves time).
2. Convert each frame to HSV color space.
3. Compute a color histogram for each frame (H: 50 bins, S: 60 bins).
4. Compare consecutive histograms using cv2.compareHist() with HISTCMP_CORREL (correlation).
5. If correlation drops below threshold (0.7), mark a scene boundary.
6. Apply segment rules:
   a. If a segment < 2 seconds → merge with the shorter neighbor.
   b. If a segment > 30 seconds → split into equal windows of ~10 seconds.
   c. If total scene count < (video_duration / 30) → too few scenes detected
      (e.g., a static lecture). Fall back to fixed 10-second windows.
```

**Why histogram correlation:**
- Fast to compute (no neural network).
- Robust to small motions (camera shake, person moving slightly).
- Sensitive to actual scene changes (cuts, transitions).
- The 0.7 threshold is standard in literature. We make it configurable.

**Why HSV not RGB:**
- HSV separates color (hue) from brightness (value).
- Makes detection robust to lighting changes within the same scene.

### 3.2 CLIP Embedding & Cosine Similarity

**How CLIP works (conceptual):**
- CLIP has two encoders: an image encoder (Vision Transformer) and a text encoder (Transformer).
- Both encoders map their input to the SAME 512-dimensional vector space.
- During training, matching image-text pairs were pushed to have HIGH cosine similarity, and mismatched pairs pushed to have LOW similarity.
- At inference: encode a frame → get a vector. Encode a query → get a vector. Cosine similarity tells you how well they match.

**Our embedding strategy (per segment):**
1. Extract 3 frames: at 10%, 50%, and 90% of the segment duration.
2. Preprocess each frame (resize to 224×224, normalize with CLIP's mean/std).
3. Encode all 3 frames → three 512-dim vectors.
4. L2-normalize each vector.
5. Average the 3 vectors → one 512-dim segment vector.
6. L2-normalize the average again.

**Why average 3 frames (not 1, not all):**
- 1 frame: Might catch a non-representative moment (transition, blur).
- All frames: Too slow, too much GPU memory.
- 3 frames (start/middle/end): Captures the visual range of the segment. Good trade-off.
- We average (not max-pool) because averaging smooths out outlier frames.

### 3.3 Reciprocal Rank Fusion (RRF)

**Problem:** We have two ranked lists — one from visual search, one from speech search. We need to combine them into one final ranked list.

**Why RRF (not weighted score averaging):**
- Visual scores (cosine similarity from CLIP) and text scores (cosine similarity from MiniLM) are on different scales and distributions.
- You can't just average them — a 0.3 from CLIP doesn't mean the same thing as a 0.3 from MiniLM.
- RRF uses only the RANK position, not the raw score. This makes it scale-invariant.
- RRF requires NO tuning. No weight hyperparameters to fiddle with.

**Algorithm:**

```
RRF_score(segment) = Σ  1 / (k + rank_in_list_i)
                     i

Where:
- k = 60 (standard constant from the original RRF paper by Cormack, Clarke & Buettcher, 2009)
- rank_in_list_i = the segment's rank in the i-th result list (1-based)
- If a segment doesn't appear in a list, it gets rank = (list_length + 1)

Final output: sort all segments by descending RRF_score, return top K.
```

**Example:**
- Segment A is rank 1 in visual, rank 5 in speech → RRF = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318
- Segment B is rank 3 in visual, rank 1 in speech → RRF = 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
- Segment B wins (appeared at top of both lists).

### 3.4 Voice Activity Detection (Silero VAD inside faster-whisper)

**Problem:** Whisper hallucinates text on silent audio (it "hears" words in noise/silence).

**Solution:** `faster-whisper` integrates Silero VAD — a small neural network (~1 MB) that classifies each audio frame as speech or non-speech. When `vad_filter=True`:
1. Audio is first scanned by Silero VAD.
2. Only segments containing speech are sent to Whisper.
3. Silent gaps are skipped entirely.

This prevents hallucinated text AND speeds up transcription (less audio to process).

### 3.5 Vector Normalization & Inner Product Search

**Why we L2-normalize all vectors:**
- Cosine similarity = dot product of L2-normalized vectors.
- FAISS `IndexFlatIP` computes dot products (inner product).
- By normalizing before storage, `IndexFlatIP` gives us cosine similarity directly.
- This is faster than using `IndexFlatL2` and converting distances.

```python
# Normalize a vector
vector = vector / np.linalg.norm(vector)

# Now: np.dot(a, b) == cosine_similarity(a, b)
```

---

## 4. Data Flow Diagrams

### 4.1 Indexing Pipeline (when `index.add("video.mp4")` is called)

```
video.mp4
    │
    ▼
┌────────────────────┐
│  1. VALIDATE       │  reader.py
│  - File exists?    │  - OpenCV: read fps, resolution, duration
│  - Is video?       │  - SHA-256 hash for dedup check
│  - Already indexed?│  - If hash exists in SQLite → skip
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  2. SCENE DETECT   │  scenes.py
│  - Sample frames   │  - Every 0.5s, convert to HSV
│  - Histogram comp  │  - Compare consecutive histograms
│  - Apply rules     │  - Min 2s, max 30s, fallback 10s windows
│  OUTPUT: list of   │
│  (start, end) times│
└────────┬───────────┘
         │
         ▼
┌────────────────────────────┐
│  3. VISUAL EMBEDDING       │  clip_encoder.py + memory.py
│  🔴 LOAD CLIP MODEL       │
│  - For each segment:      │
│    - Extract 3 frames     │  reader.py (seek to timestamp)
│    - Preprocess (224×224)  │
│    - Encode → 512-dim vec │
│    - Average → 1 vec/seg  │
│  🔴 UNLOAD CLIP MODEL     │
│  🔴 torch.cuda.empty_cache│
│  OUTPUT: N × 512 matrix   │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  4. SPEECH TRANSCRIPTION   │  asr.py + memory.py
│  - FFmpeg: extract audio   │  video.mp4 → temp.wav (16kHz mono)
│  🔴 LOAD WHISPER MODEL    │
│  - Transcribe with VAD    │
│  - Drop low-confidence    │  avg_logprob < -1.0
│  🔴 UNLOAD WHISPER MODEL  │
│  🔴 torch.cuda.empty_cache│
│  OUTPUT: list of           │
│  (start, end, text) chunks │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  5. ALIGN TRANSCRIPT       │  (in asr.py or index.py)
│  TO SEGMENTS               │
│  - For each segment:      │
│    - Find transcript chunks│
│      that overlap [start,  │
│      end]                  │
│    - Concatenate their text│
│  OUTPUT: text per segment  │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  6. TEXT EMBEDDING          │  text_encoder.py
│  🟢 LOAD MiniLM (on CPU)  │  (small enough for CPU)
│  - Encode each segment's  │
│    transcript text → 384d  │
│  - Skip segments with     │
│    empty transcript        │
│  🟢 UNLOAD MiniLM         │
│  OUTPUT: M × 384 matrix   │
│  (M ≤ N, only segments    │
│   with speech)             │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  7. STORE EVERYTHING       │  vectors.py + metadata.py
│  - Add visual vecs to     │
│    visual.faiss            │
│  - Add text vecs to       │
│    text.faiss              │
│  - Insert video row in    │
│    SQLite                  │
│  - Insert segment rows    │
│    with transcript text    │
│  - Write/update index.json│
└────────────────────────────┘
```

### 4.2 Search Pipeline (when `index.search("query", mode="both")` is called)

```
"where is backpropagation explained?"
    │
    ▼
┌──────────────────────────────┐
│  1. EMBED QUERY (VISUAL)     │  clip_encoder.py
│  🔴 LOAD CLIP (text only)   │
│  - Tokenize query           │
│  - Encode → 512-dim vec     │
│  🔴 UNLOAD CLIP             │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  2. VISUAL SEARCH            │  searcher.py + vectors.py
│  - FAISS inner product search│
│    on visual.faiss           │
│  - Get top 2K segment IDs   │
│    + scores                  │
│  OUTPUT: ranked list A       │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  3. EMBED QUERY (TEXT)       │  text_encoder.py
│  🟢 LOAD MiniLM (CPU)       │
│  - Encode query → 384-dim   │
│  🟢 UNLOAD MiniLM           │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  4. TEXT SEARCH              │  searcher.py + vectors.py
│  - FAISS inner product search│
│    on text.faiss             │
│  - Get top 2K segment IDs   │
│    + scores                  │
│  OUTPUT: ranked list B       │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  5. RECIPROCAL RANK FUSION   │  fusion.py
│  - Merge list A + list B    │
│  - RRF score per segment    │
│  - Sort descending          │
│  - Return top K             │
│  OUTPUT: final ranked list   │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  6. BUILD RESULT OBJECTS     │  searcher.py + metadata.py
│  - Look up segment metadata │
│    from SQLite               │
│  - Build SearchResult objects│
│    with video, start, end,  │
│    timestamp, score, text,  │
│    source                    │
│  OUTPUT: List[SearchResult]  │
└────────┬─────────────────────┘
         │
         ▼
    User gets results
```

---

## 5. On-Disk Index Format

When the user creates `VideoIndex("my_index")`, a folder `my_index/` is created:

```
my_index/
├── index.json          # schema version, model names, creation date
├── meta.sqlite         # videos table + segments table
├── visual.faiss        # FAISS index with CLIP vectors (512-dim)
└── text.faiss          # FAISS index with MiniLM vectors (384-dim)
```

**`index.json` format:**
```json
{
    "schema_version": 1,
    "clip_model": "ViT-B-32",
    "clip_pretrained": "laion2b_s34b_b79k",
    "whisper_model": "base",
    "text_model": "all-MiniLM-L6-v2",
    "visual_dim": 512,
    "text_dim": 384,
    "created_at": "2026-11-05T19:00:00Z"
}
```

**Rule:** On load, check `schema_version`. If it doesn't match the code's expected version, raise `IndexVersionError` with a clear message. Never silently load an incompatible index.

---

## 6. Memory Management Strategy

**The 4 GB VRAM budget:**

| Stage               | Model Loaded      | Approx VRAM (fp16/int8) | Notes                          |
|----------------------|-------------------|-------------------------|--------------------------------|
| Scene detection      | None              | 0 MB                    | CPU only (OpenCV histograms)   |
| Visual embedding     | CLIP ViT-B/32     | ~600 MB                 | fp16 on GPU                    |
| Speech transcription | Whisper base       | ~500 MB                 | int8 on GPU via CTranslate2    |
| Text embedding       | MiniLM-L6-v2      | ~80 MB                  | CPU only (deliberate choice)   |
| Vector search        | None (FAISS CPU)   | 0 MB                    | CPU only                       |

**Memory rules implemented in `models/memory.py`:**

```python
def unload_model(model) -> None:
    """Delete model, move to CPU first if needed, then free GPU cache."""
    # 1. Delete the Python reference
    del model
    # 2. Run garbage collection
    import gc
    gc.collect()
    # 3. Free CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_gpu_memory() -> dict:
    """Return total, allocated, and free GPU memory in MB."""
    ...

def auto_batch_size(base_size: int, frame_mem_mb: float = 50.0) -> int:
    """Calculate safe batch size based on available GPU memory."""
    ...
```

**OOM retry strategy:**
```
1. Try with batch_size = config default (e.g., 32).
2. If torch.cuda.OutOfMemoryError → halve batch size, retry.
3. If batch_size reaches 1 and still OOM → fall back to CPU with a warning.
```

---

## Phase 0 — Setup and Decisions

**Goal:** A clean repo where `pip install -e .` works and `pytest` runs.

### Step 0.1 — `pyproject.toml`

Create a complete `pyproject.toml` with:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "videosdk"           # CHECK PyPI FIRST — owner task
version = "0.1.0"
description = "Search inside videos by what is said and what is seen. Fully offline."
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Your Name"}
]
keywords = ["video", "search", "clip", "whisper", "offline"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Topic :: Multimedia :: Video",
]
dependencies = [
    "opencv-python-headless>=4.8,<5",
    "open-clip-torch>=2.24,<3",
    "faster-whisper>=1.0,<2",
    "sentence-transformers>=2.2,<4",
    "faiss-cpu>=1.7,<2",
    "typer>=0.9,<1",
    "tqdm>=4.60",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7",
    "pytest-cov>=4",
    "ruff>=0.4",
    "mypy>=1.8",
]

[project.scripts]
videosdk = "videosdk.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/videosdk"]

[tool.ruff]
target-version = "py310"
src = ["src"]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

### Step 0.2 — `src/videosdk/__init__.py`

```python
"""VideoSDK — Search inside videos by what is said and what is seen."""

__version__ = "0.1.0"

# Public API will be exported here in Phase 4:
# from videosdk.index import VideoIndex
# from videosdk.types import SearchResult
```

### Step 0.3 — `src/videosdk/config.py`

Define all configuration constants in one place:

```python
"""Central configuration — all magic numbers and model names live here."""

from dataclasses import dataclass, field

@dataclass
class Config:
    # Scene detection
    scene_hist_threshold: float = 0.7
    scene_sample_interval: float = 0.5   # seconds between sampled frames
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
```

### Step 0.4 — `src/videosdk/errors.py`

```python
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
            "  Windows: winget install FFmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )

class IndexVersionError(VideoSDKError):
    """Raised when the on-disk index version doesn't match."""
    def __init__(self, found: int, expected: int) -> None:
        super().__init__(
            f"Index version mismatch: found v{found}, expected v{expected}. "
            f"Re-index your videos with the current version."
        )

class InvalidVideoError(VideoSDKError):
    """Raised when a file cannot be read as a video."""
    def __init__(self, path: str, reason: str = "") -> None:
        msg = f"Cannot read as video: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)
```

### Step 0.5 — `src/videosdk/types.py`

```python
"""Typed data objects returned by the public API."""

from dataclasses import dataclass

@dataclass
class SearchResult:
    """A single search result pointing to a moment in a video."""
    video: str          # path of the source video
    start: float        # start time in seconds
    end: float          # end time in seconds
    timestamp: str      # human-readable "MM:SS" or "H:MM:SS"
    score: float        # higher is better
    text: str           # matched transcript text (may be empty)
    source: str         # "visual" | "speech" | "both"

    def clip(self, output_path: str) -> str:
        """Save a short clip of this result to output_path. Returns the path."""
        from videosdk.video.clipper import cut_clip
        return cut_clip(self.video, self.start, self.end, output_path)
```

### Step 0.6 — Smoke test in `tests/test_api.py`

```python
"""Smoke tests to verify the package imports and basic types work."""

from videosdk import __version__
from videosdk.types import SearchResult
from videosdk.config import Config
from videosdk.errors import VideoNotFoundError, IndexVersionError

def test_version_exists():
    assert __version__ is not None
    assert isinstance(__version__, str)

def test_search_result_creation():
    r = SearchResult(
        video="test.mp4", start=10.0, end=15.0,
        timestamp="0:10", score=0.85,
        text="hello world", source="speech"
    )
    assert r.video == "test.mp4"
    assert r.start == 10.0
    assert r.source == "speech"

def test_config_defaults():
    c = Config()
    assert c.clip_model_name == "ViT-B-32"
    assert c.schema_version == 1

def test_errors():
    import pytest
    with pytest.raises(VideoNotFoundError, match="test.mp4"):
        raise VideoNotFoundError("test.mp4")
    with pytest.raises(IndexVersionError, match="v2.*v1"):
        raise IndexVersionError(found=2, expected=1)
```

### Step 0.7 — GitHub Actions CI (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Lint with ruff
        run: ruff check src/ tests/
      - name: Type check with mypy
        run: mypy src/videosdk/ --ignore-missing-imports
      - name: Run tests
        run: pytest --tb=short -q
```

### Step 0.8 — LICENSE (MIT)

Write a standard MIT license file.

### Step 0.9 — Validation

```
pip install -e ".[dev]"
ruff check src/ tests/
pytest -v
```

All must pass before moving to Phase 1.

### Step 0.10 — Update TRACKER.md and notes.md

Tick Phase 0 checkboxes in `TRACKER.md`. Log environment versions. Add session log entry. Update `notes.md` with Phase 0 setup notes and memory management strategy.

---

## Phase 1 — Visual Search Prototype

**Goal:** Type a text query → get ranked timestamps by visual content.

### Step 1.1 — `src/videosdk/video/reader.py`

**What to implement:**
- `get_video_metadata(path) -> VideoMeta`: Opens video with `cv2.VideoCapture`, reads `CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`, `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT`. Calculates duration. Returns a typed dataclass.
- `read_frame_at_time(cap, time_sec) -> np.ndarray`: Seeks to a timestamp using `cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)`, reads one frame, converts BGR→RGB. Returns the frame as a numpy array.
- `iter_frames_at_times(path, times) -> Iterator[np.ndarray]`: Opens video, yields frames at a list of timestamps. Closes video when done. Never holds more than one frame in memory.
- `compute_file_hash(path) -> str`: SHA-256 hash of the file for deduplication.

**Key implementation details:**
- Always check `cap.isOpened()` and `ret` (from `cap.read()`) — raise `InvalidVideoError` on failure.
- Use `cv2.CAP_PROP_POS_MSEC` for seeking (millisecond precision), not frame numbers.
- Convert BGR→RGB immediately (CLIP expects RGB).

### Step 1.2 — `src/videosdk/video/scenes.py`

**What to implement:**
- `detect_scenes(path, config) -> list[tuple[float, float]]`: The full scene detection algorithm from Section 3.1.
- `_compute_histograms(path, sample_interval) -> list[tuple[float, np.ndarray]]`: Sample frames and compute HSV histograms.
- `_find_boundaries(histograms, threshold) -> list[float]`: Compare consecutive histograms, return timestamps where scenes change.
- `_apply_segment_rules(boundaries, duration, config) -> list[tuple[float, float]]`: Enforce min/max/fallback rules.

**Implementation pseudocode for `_apply_segment_rules`:**
```
1. Convert boundary timestamps to (start, end) segments.
2. Merge pass: walk segments, if segment.duration < min_seconds, merge with shorter neighbor.
3. Split pass: walk segments, if segment.duration > max_seconds, split into equal parts ≈ fallback_seconds.
4. Fallback check: if len(segments) < duration / 30, replace everything with fixed windows of fallback_seconds.
5. Return final segment list.
```

### Step 1.3 — `src/videosdk/models/memory.py`

**What to implement:**
- `unload_model(model)`: As described in Section 6.
- `get_gpu_memory() -> dict[str, float]`: Returns `{"total_mb": ..., "allocated_mb": ..., "free_mb": ...}`.
- `get_device(preference) -> torch.device`: Resolves `"auto"` / `"cuda"` / `"cpu"` to an actual device. `"auto"` → cuda if available, else cpu.
- `oom_retry(fn, *args, batch_size, min_batch, shrink_factor)`: Decorator/helper that catches `torch.cuda.OutOfMemoryError`, halves batch size, retries, falls back to CPU.

### Step 1.4 — `src/videosdk/models/clip_encoder.py`

**What to implement:**
- `CLIPEncoder` class:
  - `__init__(self, config, device)`: Stores config, does NOT load model yet.
  - `load(self)`: Calls `open_clip.create_model_and_transforms()`, loads model to device, sets `model.eval()`, enables fp16 if on CUDA.
  - `encode_frames(self, frames: list[np.ndarray]) -> np.ndarray`: Preprocesses frames, runs through image encoder in batches, L2-normalizes, returns matrix.
  - `encode_text(self, text: str) -> np.ndarray`: Tokenizes, runs through text encoder, L2-normalizes, returns 1×512 vector.
  - `unload(self)`: Calls `memory.unload_model()`.
- `embed_segments(path, segments, config, device) -> np.ndarray`: High-level function. Creates encoder, loads it, for each segment extracts 3 frames, encodes them, averages, unloads encoder. Returns N×512 matrix.

**Key details:**
- Use `torch.no_grad()` and `torch.cuda.amp.autocast()` for fp16.
- Batch frames across segments (not per-segment) for GPU efficiency.
- L2-normalize EVERY vector before storing.

### Step 1.5 — `src/videosdk/store/vectors.py`

**What to implement:**
- `VectorStore` class:
  - `__init__(self, path, dim)`: Path to the `.faiss` file, dimensionality.
  - `create(self)`: Creates a new `faiss.IndexFlatIP(dim)`.
  - `load(self)`: Loads existing index from disk.
  - `add(self, vectors: np.ndarray) -> int`: Adds vectors, returns starting row ID.
  - `search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]`: Returns (scores, indices).
  - `save(self)`: Writes index to disk.
  - `count(self) -> int`: Returns number of vectors in the index.

### Step 1.6 — Write and pass `tests/test_scenes.py`

**Test cases:**
- Test segment merge (segments shorter than 2s get merged).
- Test segment split (segments longer than 30s get split).
- Test fallback to fixed windows when too few scenes detected.
- Test with synthetic histogram data (no real video needed for unit tests).

### Step 1.7 — Integration test script

Create a plain script (`examples/visual_search_demo.py` or a test) that:
1. Takes a video path and a query string.
2. Runs scene detection → visual embedding → FAISS search.
3. Prints top-5 timestamps with scores.

**Validate on a real video.** Log peak GPU memory in `TRACKER.md` and update `notes.md` with complete Phase 1 visual search notes, key code implementations, and performance results.

---

## Phase 2 — Speech Search

**Goal:** Search by what is said in the video.

### Step 2.1 — Audio extraction utility

In `src/videosdk/video/reader.py` add:
- `extract_audio(video_path, output_path, sample_rate=16000)`:
  Calls FFmpeg: `ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav`
  Uses `subprocess.run()` with timeout. Raises `FFmpegNotFoundError` if ffmpeg not found.

### Step 2.2 — `src/videosdk/models/asr.py`

**What to implement:**
- `WhisperTranscriber` class:
  - `__init__(self, config, device)`: Stores config. Does NOT load model yet.
  - `load(self)`: Creates `faster_whisper.WhisperModel(size, device, compute_type)`.
  - `transcribe(self, audio_path) -> list[TranscriptChunk]`:
    - Runs `model.transcribe(audio_path, vad_filter=True)`.
    - Collects segments with `start`, `end`, `text`, `avg_logprob`.
    - Drops segments where `avg_logprob < config.whisper_min_confidence`.
    - Returns typed list.
  - `unload(self)`: Calls `memory.unload_model()`.

- `TranscriptChunk` dataclass: `start: float, end: float, text: str, confidence: float`

- `align_transcript_to_segments(chunks, segments) -> dict[int, str]`:
  For each segment (by index), find all transcript chunks whose time range overlaps with the segment's time range. Concatenate their text. Return `{segment_index: text}`.

  **Overlap logic:**
  ```python
  def overlaps(chunk_start, chunk_end, seg_start, seg_end) -> bool:
      return chunk_start < seg_end and chunk_end > seg_start
  ```

### Step 2.3 — `src/videosdk/models/text_encoder.py`

**What to implement:**
- `TextEncoder` class:
  - `__init__(self, config)`: Stores config. Deliberately always uses CPU.
  - `load(self)`: Creates `SentenceTransformer(model_name, device="cpu")`.
  - `encode(self, texts: list[str]) -> np.ndarray`: Encodes texts in batches, L2-normalizes, returns M×384 matrix.
  - `unload(self)`: Deletes model.

### Step 2.4 — `src/videosdk/store/metadata.py`

**What to implement:**
- `MetadataStore` class:
  - `__init__(self, db_path)`: Path to SQLite file.
  - `create_tables(self)`: Runs the CREATE TABLE statements from Section 2.7.
  - `add_video(self, path, file_hash, duration, fps, width, height) -> int`: INSERT, return video_id.
  - `add_segments(self, video_id, segments_data)`: Bulk INSERT segments.
  - `get_video(self, path) -> dict | None`: Look up by path.
  - `get_video_by_hash(self, file_hash) -> dict | None`: Look up by hash.
  - `get_segment(self, segment_id) -> dict`: Look up one segment.
  - `get_segments_for_video(self, video_id) -> list[dict]`: All segments for a video.
  - `get_segment_by_index(self, index_id) -> dict`: Look up by FAISS row index.
  - `list_videos(self) -> list[dict]`: All indexed videos.
  - `remove_video(self, path)`: DELETE video + CASCADE segments.
  - `has_video_hash(self, file_hash) -> bool`: Dedup check.

### Step 2.5 — Write and pass `tests/test_store.py`

**Test cases:**
- Create tables, add a video, add segments, query them back.
- Dedup check: same hash should be detected.
- Remove video cascades to segments.
- Store uses a temporary SQLite file (use `tmp_path` fixture).

### Step 2.6 — Validate speech pipeline end-to-end

Test with:
- A video with clear speech → transcript should be accurate.
- A video with long silences → VAD should skip them, no hallucination.
- A video with NO speech → should return empty transcript, not crash.

Log Whisper timing and peak memory in `TRACKER.md` and update `notes.md` with Phase 2 speech search notes, Whisper configuration, and transcription pipeline details.

---

## Phase 3 — Combine and Rank (Fusion)

**Goal:** One ranked result list from visual + speech search.

### Step 3.1 — `src/videosdk/search/fusion.py`

**What to implement:**

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],  # each: [(segment_id, score), ...]
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each list contains (segment_id, original_score) tuples,
                      ordered by descending score (best first).
        k: RRF constant (default 60 from the original paper).

    Returns:
        Combined list of (segment_id, rrf_score) sorted by descending rrf_score.
    """
    rrf_scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (seg_id, _original_score) in enumerate(ranked_list, start=1):
            rrf_scores[seg_id] = rrf_scores.get(seg_id, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

### Step 3.2 — `src/videosdk/search/searcher.py`

**What to implement:**
- `Searcher` class:
  - `__init__(self, visual_store, text_store, metadata_store, config)`.
  - `search(self, query, top_k, mode, device) -> list[SearchResult]`:
    - If mode == "visual": only visual search.
    - If mode == "speech": only text search.
    - If mode == "both": both → RRF fusion.
    - Builds `SearchResult` objects from metadata.
  - `_visual_search(self, query, top_k, device) -> list[tuple[int, float]]`:
    Load CLIP text encoder, encode query, search visual FAISS, unload.
  - `_speech_search(self, query, top_k, device) -> list[tuple[int, float]]`:
    Load MiniLM, encode query, search text FAISS, unload.

**Timestamp formatting helper:**
```python
def format_timestamp(seconds: float) -> str:
    """Convert 754.5 → '12:34' or 3754.5 → '1:02:34'."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
```

### Step 3.3 — Write and pass `tests/test_fusion.py`

**Test cases (pure logic, no models needed):**

```python
def test_rrf_basic():
    list_a = [(1, 0.9), (2, 0.8), (3, 0.7)]
    list_b = [(2, 0.95), (3, 0.85), (1, 0.6)]
    result = reciprocal_rank_fusion([list_a, list_b], k=60)
    # Segment 2: rank 2 in A, rank 1 in B → highest RRF
    assert result[0][0] == 2

def test_rrf_single_list():
    list_a = [(5, 0.9), (3, 0.5)]
    result = reciprocal_rank_fusion([list_a], k=60)
    assert result[0][0] == 5

def test_rrf_empty():
    result = reciprocal_rank_fusion([], k=60)
    assert result == []

def test_rrf_no_overlap():
    list_a = [(1, 0.9), (2, 0.8)]
    list_b = [(3, 0.95), (4, 0.85)]
    result = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert len(result) == 4
```

### Step 3.4 — Update TRACKER.md and notes.md

Verify all fusion modes and tests pass. Update `TRACKER.md` and document RRF fusion mechanics, reciprocal scoring formulas, and results in `notes.md`.

---

## Phase 4 — Turn It Into an SDK

**Goal:** The public API from `PROJECT_OVERVIEW.md` Section 5 works exactly as written.

### Step 4.1 — `src/videosdk/index.py` — The `VideoIndex` class

This is the main orchestrator. It ties everything together.

```python
class VideoIndex:
    def __init__(self, path: str, device: str = "auto"):
        """Create or open an index stored in a folder."""
        # Resolve device
        # If path exists → load existing index (check schema_version)
        # If path doesn't exist → create new index folder + empty stores

    def add(self, video_path: str) -> None:
        """Add a video to the index. Runs the full indexing pipeline."""
        # 1. Validate video file
        # 2. Check dedup (file hash in metadata store)
        # 3. Scene detection → segments
        # 4. Visual embedding (load CLIP → embed → unload)
        # 5. Audio extraction → speech transcription (load Whisper → transcribe → unload)
        # 6. Align transcript to segments
        # 7. Text embedding (load MiniLM → embed → unload)
        # 8. Store vectors + metadata
        # 9. Save everything to disk
        # All with tqdm progress bars

    def search(self, query: str, top_k: int = 5, mode: str = "both") -> list[SearchResult]:
        """Search the index. Returns ranked results."""
        # Delegates to Searcher

    def list_videos(self) -> list[str]:
        """List all indexed video paths."""

    def remove(self, video_path: str) -> None:
        """Remove a video from the index."""
        # Remove from metadata
        # Rebuild FAISS indexes (FAISS doesn't support deletion natively
        # so we rebuild from remaining vectors)
```

### Step 4.2 — `src/videosdk/video/clipper.py`

```python
def cut_clip(video_path: str, start: float, end: float, output_path: str) -> str:
    """Cut a clip from a video using FFmpeg stream copy."""
    # ffmpeg -y -ss {start} -to {end} -i {video_path} -c copy {output_path}
    # Validate inputs, check FFmpeg exists, run subprocess
    # Return output_path
```

### Step 4.3 — `src/videosdk/cli.py`

```python
import typer
app = typer.Typer(name="videosdk", help="Search inside videos.")

@app.command()
def add(index_path: str, video_path: str, device: str = "auto"):
    """Add a video to an index."""

@app.command()
def search(index_path: str, query: str, top_k: int = 5, mode: str = "both"):
    """Search an index."""

@app.command()
def list(index_path: str):
    """List videos in an index."""

@app.command()
def remove(index_path: str, video_path: str):
    """Remove a video from an index."""

@app.command()
def doctor():
    """Check system dependencies and hardware."""
    # Check Python version
    # Check FFmpeg
    # Check GPU (torch.cuda.is_available, memory)
    # Check disk space
    # Print all in a nice table
```

### Step 4.4 — Update `__init__.py` exports

```python
from videosdk.index import VideoIndex
from videosdk.types import SearchResult

__all__ = ["VideoIndex", "SearchResult", "__version__"]
```

### Step 4.5 — Write and pass `tests/test_api.py` (expanded)

Test the full public API contract from Section 5 of `PROJECT_OVERVIEW.md`:
- `VideoIndex` creates a folder.
- `add()` with a tiny test video works.
- `search()` returns `SearchResult` objects with correct fields.
- `list_videos()` returns the added video.
- `remove()` removes it.
- Error cases: missing file, invalid video, missing FFmpeg.

### Step 4.6 — Update TRACKER.md and notes.md

Verify CLI and public API pass all integration tests. Update `TRACKER.md` and document SDK usage patterns, index directory schemas, and CLI commands in `notes.md`.

---

## Phase 4.5 — Advanced Multimodal Intelligence & Creator Tools

**Goal:** Elevate VideoSDK beyond basic text-to-video search into an industry-grade, multimodal intelligence engine with reverse visual search, automated highlight reels, video editor format exports (SRT/EDL/FCPXML), smart chapters, non-speech audio events, and an embedded zero-node local Web UI.

### Architectural Positioning
All Phase 4.5 capabilities build directly on the public `VideoIndex` orchestrator and core primitives without breaking backward compatibility or exceeding the **4 GB VRAM ceiling**:
- Reverse visual search reuses the existing OpenCLIP vision model pipeline.
- Highlight generation leverages the stream-copy and filter graph infrastructure in `clipper.py`.
- Exporters are pure-Python transformation utilities without neural network overhead.
- Smart chaptering and heatmaps use existing segment vectors and cosine similarity curves.
- Sound event search introduces optional CLAP audio embedding under strict sequential memory lifecycle rules.
- The Web UI provides a standalone local browser interface (`videosdk ui`) using Gradio/Streamlit.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Creator Tools & Advanced Layer                         │
│   Reverse Visual Search  │  Highlight Reels  │  Editor Exports  │  Web UI (Gradio)│
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────────┐
│                              VideoIndex API (index.py)                          │
│ search(image=...) │ create_supercut() │ export_results() │ generate_chapters() │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌──────────────────┬─────────────────────┼─────────────────────┬──────────────────┐
│  CLIP Encoder    │  Whisper ASR + MiniLM│ CLAP Audio Events   │ FFmpeg Clipper   │
│ (Image & Text)   │  (Spoken Words)     │ (Sound Effects)     │ & Reel Concat    │
└──────────────────┴─────────────────────┴─────────────────────┴──────────────────┘
```

---

### Step 4.5.1 — Reverse Visual Search (Image-to-Video Retrieval)

**Module:** `src/videosdk/index.py` & `src/videosdk/models/clip_encoder.py`

**What it does:** Allows users to query the index using an image (file path, URL, or PIL/NumPy image) to find every scene where that object, person, logo, landmark, or visual style appears across all indexed videos.

**API Contract:**
```python
# Text search (existing)
results = index.search(query="vintage sports car drifting")

# Image-to-video search (new)
results = index.search(image="path/to/reference_photo.jpg", top_k=5)

# Combined multi-modal search (text + visual reference)
results = index.search(query="in the mountains", image="car.jpg", top_k=5)
```

**Implementation Details:**
1. Update `ClipEncoder.encode_query(query: str | Image.Image | Path | np.ndarray) -> np.ndarray`:
   - If string and exists as a file path, load via PIL: `Image.open(file_path).convert("RGB")`.
   - Preprocess through `clip_encoder.preprocess(img).unsqueeze(0)`.
   - Pass through `clip_encoder.model.encode_image(tensor)` and L2-normalize (`vec / norm`).
   - If both `query` and `image` are provided, compute both embeddings, average them with equal or customizable weight, and re-normalize before searching the visual FAISS index.
2. Update `VideoIndex.search` and CLI `videosdk search --image path/to/image.jpg`.

---

### Step 4.5.2 — Automated Supercut / Highlight Reel Generator

**Module:** `src/videosdk/video/clipper.py` & `src/videosdk/index.py`

**What it does:** Takes search query results or specific topic matches and automatically compiles, trims, and stitches them into a single continuous video reel (e.g. "top 60 seconds of goal celebrations" or "summary of keynote announcements").

**API Contract:**
```python
supercut_path = index.create_supercut(
    query="standing ovation",
    output_path="applause_reel.mp4",
    max_duration=60.0,       # Total max duration in seconds
    clip_padding=0.5,        # Seconds of lead-in/lead-out padding
    merge_threshold=2.0,     # Merge adjacent clips if within 2 seconds
    transition="crossfade",  # None | "cut" | "crossfade"
)
```

**Implementation Details:**
1. Run `search(query, top_k=20)`.
2. Extract time ranges `[(start, end)]`.
3. Apply temporal merging: if two candidate segments overlap or are within `merge_threshold` seconds of each other in the same video, merge them into a single interval `(start_1, end_2)` to prevent choppy jumps.
4. Cumulative duration budgeting: sort by score, greedily select clips until `sum(durations) <= max_duration`, then re-sort selected clips chronologically per video.
5. FFmpeg concat pipeline:
   - Cut individual trimmed clips to a temporary scratch directory.
   - For fast cuts (`transition="cut"`): write `concat.txt` manifest (`file 'clip_0.mp4' ...`) and invoke `ffmpeg -f concat -safe 0 -i concat.txt -c copy {output_path}`.
   - For smooth crossfades: construct an FFmpeg filter graph `[0:v][1:v]xfade=transition=fade:duration=0.5[v]` to blend audio and video seamlessly.
   - Clean up intermediate temporary files upon completion or failure.

---

### Step 4.5.3 — Professional Video Editor & Subtitle Exporters

**Module:** `src/videosdk/export/exporters.py`

**What it does:** Allows video editors, researchers, and content creators to export search hits directly into industry-standard NLE (Non-Linear Editor) marker files and subtitle tracks.

**Supported Formats:**
1. **SRT (`.srt`) & WebVTT (`.vtt`)**:
   - Creates timed subtitle entries with search score and transcribed speech or visual label for display in VLC, YouTube, or web video players.
2. **EDL (Edit Decision List - CMX 3600)**:
   - Imports directly into DaVinci Resolve, Adobe Premiere Pro, and Final Cut Pro as timeline cuts.
3. **FCPXML (Final Cut Pro XML)**:
   - Imports into Final Cut Pro and DaVinci Resolve as colored sequence markers with search query notes.
4. **CSV / JSON Cue Sheets**:
   - Structured tabular exports for spreadsheets, archival databases, and production logs.

**API Contract:**
```python
from videosdk.export import export_results

results = index.search("explosion scene", top_k=10)

# Export to subtitle file
export_results(results, format="srt", output_path="scenes.srt")

# Export to DaVinci Resolve / Premiere marker list
export_results(results, format="edl", output_path="timeline.edl")

# Export to structured CSV
export_results(results, format="csv", output_path="metadata.csv")
```

---

### Step 4.5.4 — Smart Video Chapters & Timeline Heatmaps

**Module:** `src/videosdk/analytics/chapters.py` & `src/videosdk/analytics/heatmap.py`

**What it does:**
1. **Automated Chaptering**: Segments long-form videos (lectures, podcasts, webinars) into logical chapters with start timestamps, durations, and AI-summarized topic headers using transcript clustering and visual scene boundary density.
2. **Timeline Relevance Heatmaps**: Computes a continuous similarity score (0.0 to 1.0) sampled every N seconds across the entire video timeline for a given query, enabling YouTube-style scrub bar engagement curves.

**API Contract:**
```python
# Generate structured video chapters
chapters = index.generate_chapters(
    video_path="lecture.mp4",
    min_chapter_duration=60.0,  # Minimum 1 minute per chapter
)
for ch in chapters:
    print(f"[{ch.timestamp}] {ch.title} ({ch.duration:.1f}s)")

# Generate scrub bar heatmap curve
heatmap = index.generate_heatmap(
    video_path="gameplay.mp4",
    query="intense battle",
    step_seconds=2.0,
)
# Returns list of dicts: [{"time": 0.0, "score": 0.12}, {"time": 2.0, "score": 0.89}, ...]
```

---

### Step 4.5.5 — Non-Speech Audio / Sound Event Retrieval (CLAP)

**Module:** `src/videosdk/models/audio_event.py`

**What it does:** Extends speech search to non-verbal audio events: music genres, gunshots, screams, glass breaking, clapping/applause, car engines, animal sounds, laughing, and rain.

**Architecture:**
- Uses a lightweight audio-language contrastive model (e.g. `laion/larger_clap_general` or quantized HTSAT-CLAP).
- Audio is extracted once into 16kHz mono WAV during Phase 2 ingestion.
- While Whisper transcribes verbal speech, the CLAP encoder splits audio into 5-second overlapping windows to compute 512-dim audio event embeddings.
- Stored in a third FAISS index: `audio_events.index`.
- When users search with `mode="sound"` or `mode="all"`, Reciprocal Rank Fusion combines visual (CLIP), speech (Whisper+MiniLM), and sound events (CLAP).
- **Memory Safety:** Loaded strictly sequentially; Whisper is purged before CLAP initializes. CLAP is purged before MiniLM initializes.

---

### Step 4.5.6 — Embedded Zero-Node Local Web UI (`videosdk ui`)

**Module:** `src/videosdk/ui/app.py` & `src/videosdk/cli.py`

**What it does:** Provides an out-of-the-box, zero-configuration web interface for visual and spoken video exploration running locally without needing Node.js, npm, or complex frontend builds.

**CLI Command:**
```bash
videosdk ui --index ./my_video_index --port 7860
```

**Features:**
1. **Interactive Search Bar**: Supports text search, image drag-and-drop search, and filter mode toggles (Visual / Speech / Hybrid).
2. **Video Player with Deep Linking**: Clicking any search result card instantly seeks the HTML5 video player to the exact start timestamp.
3. **Synchronized Transcript Sidebar**: Displays the aligned segment transcript with highlighted search keywords.
4. **Instant Clip Export**: Single-click button to download the exact 10-second segment or trigger supercut creation.
5. **System Telemetry Banner**: Displays active index statistics (indexed hours, segment count, disk size) and current VRAM usage.

---

### Step 4.5.7 — Unit & Integration Test Specifications

Create `tests/test_advanced.py`:
- `test_reverse_visual_search`: Verify PIL Image query returns matching visual segment vectors.
- `test_supercut_interval_merging`: Verify overlapping intervals `[(0, 5), (4, 10)]` merge to `[(0, 10)]`.
- `test_supercut_duration_cap`: Verify total stitched duration does not exceed `max_duration`.
- `test_export_srt`: Verify generated `.srt` conforms to SubRip timestamp spec (`00:01:23,456 --> 00:01:28,789`).
- `test_export_vtt`: Verify generated `.vtt` includes `WEBVTT` header and valid cues.
- `test_export_edl`: Verify CMX 3600 standard formatting with correct timecodes.
- `test_heatmap_generation`: Verify dense time step sampling across video duration.

### Step 4.5.8 — Update TRACKER.md and notes.md

Verify all advanced creator and multimodal features pass unit tests, update `TRACKER.md` checklist, and document technical patterns in `notes.md`.

---

## Phase 5 — Tests, Eval Set, and Benchmarks

**Goal:** Real measured numbers. Proof that it works.

### Step 5.1 — `benchmarks/run_benchmark.py`

A single script that:
1. Loads the eval set (videos + labeled queries from `benchmarks/eval_set/`).
2. Indexes all videos (if not already indexed).
3. Runs every query in all three modes (visual, speech, both).
4. Computes Recall@1, Recall@5, MRR for each mode.
5. Measures and reports:
   - Indexing time per stage per hour of video.
   - Peak GPU memory per stage.
   - Peak RAM.
   - Query time.
   - Index size on disk.
6. Saves results to `benchmarks/results/` as JSON and markdown table.

**Metrics formulas:**

```
Recall@K = (# of queries where correct segment is in top K) / (total queries)

MRR (Mean Reciprocal Rank) = (1/N) × Σ (1 / rank_of_correct_answer)
    where rank = position of the first correct result (1-based)
    if correct answer not found, 1/rank = 0
```

### Step 5.2 — CPU vs GPU comparison

Run the full benchmark twice: once with `device="cuda"`, once with `device="cpu"`.
Report both in a comparison table.

### Step 5.3 — CI integration

Add a small CPU-only test to CI that:
- Creates an index from a tiny (3-second) test video.
- Runs a search query.
- Verifies results are returned without crashing.

---

## Phase 6 — Docs, Polish, and Release

**Goal:** A stranger can install and understand it in 30 seconds.

### Step 6.1 — README.md structure

```markdown
# VideoSDK — Search Inside Videos

One-line pitch.

![Demo GIF](docs/demo.gif)

## Install
## Quick Start (5 lines of code)
## CLI Usage
## How It Works (short)
## Benchmark Results (table)
## Known Limits (honest)
## Model Licenses
## Design Choices (short write-up)
## Contributing
```

### Step 6.2 — `examples/demo.ipynb`

Jupyter notebook showing:
1. Indexing a video.
2. Searching by visual content.
3. Searching by speech.
4. Comparing results.
5. Saving a clip.

### Step 6.3 — `docs/design.md`

Short write-up:
- Why one model at a time.
- Why RRF over weighted averaging.
- Why histogram-based scene detection (not neural).
- What failed and what you'd do differently.

### Step 6.4 — PyPI release (Owner task)

1. Test on TestPyPI first.
2. `pip install` from TestPyPI on a clean venv.
3. Publish to real PyPI.
4. Test `pip install videosdk` on a fresh machine.

### Step 6.5 — Update TRACKER.md and notes.md

Complete final documentation, benchmarks, and release checklists. Update `TRACKER.md` and finalize `notes.md` with release notes and production guidelines.

---

## 15. Error Handling Strategy

**Principle:** Every error message must tell the user WHAT went wrong and HOW to fix it.

| Error Situation               | Exception                | Message Example                                         |
|-------------------------------|--------------------------|--------------------------------------------------------|
| Video file not found          | `VideoNotFoundError`     | "Video file not found: /path/to/file.mp4"              |
| File is not a valid video     | `InvalidVideoError`      | "Cannot read as video: file.txt (not a video format)"  |
| FFmpeg not installed          | `FFmpegNotFoundError`    | "FFmpeg not found. Install it: ..."                    |
| Index version mismatch        | `IndexVersionError`      | "Index version mismatch: found v2, expected v1. ..."   |
| GPU out of memory             | (handled internally)     | Warning: "GPU out of memory, falling back to CPU."     |
| Video already indexed         | (skip silently)          | Info: "Video already indexed, skipping."               |
| No speech in video            | (no error)               | Info: "No speech detected in video."                   |

---

## 16. Testing Strategy

### Unit tests (no real models, no real videos)

| Module          | Test File           | What to test                                              |
|-----------------|--------------------|------------------------------------------------------------|
| `scenes.py`     | `test_scenes.py`   | Segment merge, split, fallback rules with synthetic data   |
| `fusion.py`     | `test_fusion.py`   | RRF with hand-crafted ranked lists                         |
| `metadata.py`   | `test_store.py`    | SQLite CRUD operations with temp DB                        |
| `vectors.py`    | `test_store.py`    | FAISS add/search/save/load with random vectors             |
| `types.py`      | `test_api.py`      | SearchResult creation and fields                           |
| `errors.py`     | `test_api.py`      | All custom exceptions                                      |
| `config.py`     | `test_api.py`      | Default values                                              |

### Integration tests (need models, use tiny videos)

| Test                        | What it validates                              |
|-----------------------------|-----------------------------------------------|
| Index a 3-second video      | Full pipeline runs without crashing           |
| Search returns results      | API contract works                            |
| CPU fallback                | Works without GPU                             |
| Video with no audio track   | Graceful handling                             |
| `videosdk doctor`           | CLI runs and reports status                   |

---

> **End of Implementation Guide.**
> At each phase, open this file, follow the steps exactly, validate with the listed tests, then update `TRACKER.md` and document detailed phase notes in `notes.md`.
