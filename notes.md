# VideoSDK Project Notes & Engineering Reference

This document tracks technical decisions, architectural details, key code patterns, and practical notes across each phase of development. It serves as an ongoing learning reference and living technical guide.

---

## Table of Contents
1. [Architecture & Constraints](#architecture--constraints)
2. [Phase 0 — Setup, Environment & Memory Foundation](#phase-0--setup-environment--memory-foundation)
3. [Phase 1 — Visual Search Prototype](#phase-1--visual-search-prototype)
   - [Overview & Pipeline](#overview--pipeline)
   - [Module 1: Video Reader (`video/reader.py`)](#module-1-video-reader-videoreaderpy)
   - [Module 2: Scene Detection (`video/scenes.py`)](#module-2-scene-detection-videoscenespy)
   - [Module 3: Visual & Query Embedding (`models/clip_encoder.py`)](#module-3-visual--query-embedding-modelsclip_encoderpy)
   - [Module 4: FAISS Vector Store (`store/vectors.py`)](#module-4-faiss-vector-store-storevectorspy)
   - [End-to-End Visual Search Script (`examples/visual_search_demo.py`)](#end-to-end-visual-search-script-examplesvisual_search_demopy)
   - [Verification & Test Results](#verification--test-results)
4. [Phase 2 — Speech Search](#phase-2--speech-search)
   - [Overview & Pipeline](#overview--pipeline-1)
   - [Module 1: Audio Extraction (`video/reader.py:extract_audio`)](#module-1-audio-extraction-videoreaderpyextract_audio)
   - [Module 2: Faster-Whisper ASR & Alignment (`models/asr.py`)](#module-2-faster-whisper-asr--alignment-modelsasrpy)
   - [Module 3: Sentence-Transformers MiniLM (`models/text_encoder.py`)](#module-3-sentence-transformers-minilm-modelstext_encoderpy)
   - [Module 4: SQLite Metadata Store (`store/metadata.py`)](#module-4-sqlite-metadata-store-storemetadatapy)
   - [Verification & Test Results](#verification--test-results-1)
5. [Phase 3 — Fusion & Ranking](#phase-3--fusion--ranking)
   - [Overview & Pipeline](#overview--pipeline-2)
   - [Module 1: Reciprocal Rank Fusion (`search/fusion.py`)](#module-1-reciprocal-rank-fusion-srcvideosdksearchfusionpy)
   - [Module 2: Unified Search Coordinator (`search/searcher.py`)](#module-2-unified-search-coordinator-srcvideosdksearchsearcherpy)
   - [Verification & Test Results](#verification--test-results-2)
6. [Phase 4 — SDK Public API & CLI](#phase-4--sdk-public-api--cli-upcoming) *(Upcoming)*
7. [Phase 4.5 — Advanced Multimodal Intelligence & Creator Tools](#phase-45--advanced-multimodal-intelligence--creator-tools-upcoming) *(Upcoming)*
8. [Phase 5 — Testing & Evaluation Benchmarks](#phase-5--testing--evaluation-benchmarks-upcoming) *(Upcoming)*
9. [Phase 6 — Documentation & Distribution](#phase-6--documentation--distribution-upcoming) *(Upcoming)*

---

## Architecture & Constraints

- **Core Goal**: Offline semantic search over video files by visual content and spoken content.
- **Hardware Constraint**: Must run reliably on a single GPU with **4 GB VRAM** (e.g. NVIDIA RTX 3050 Laptop) or smoothly fall back to CPU.
- **Architectural Rule #1**: **Sequential Model Loading**. Models (OpenCLIP, faster-whisper, sentence-transformers) must **never** run in GPU memory at the same time. Each model is loaded on demand, performs its task, and is explicitly purged and garbage collected before the next model is loaded.
- **Architectural Rule #2**: **Zero Cloud Dependencies**. The entire search, indexing, embedding, and transcription pipeline runs completely offline.

---

## Phase 0 — Setup, Environment & Memory Foundation

### Key Decisions & Deliverables
1. **Packaging**: Configured `pyproject.toml` using `hatchling` backend with strict dev tooling (`ruff`, `mypy`, `pytest`).
2. **Device & Memory Manager (`src/videosdk/models/memory.py`)**:
   - `get_device(preference)`: Resolves `"auto"`, `"cuda"`, or `"cpu"`.
   - `get_gpu_memory()`: Reports GPU total, allocated, and free memory in MB.
   - `unload_model(model)`: Explicitly calls `del model`, `gc.collect()`, and `torch.cuda.empty_cache()` to reclaim VRAM.
   - `oom_retry(...)`: Catches `torch.cuda.OutOfMemoryError`, halves batch sizes, and falls back to CPU if memory is exhausted.

---

## Phase 1 — Visual Search Prototype

### Overview & Pipeline

Phase 1 allows users to input a natural-language search query (e.g., `"green forest"`, `"fire and flame"`, `"man walking on street"`) and retrieve the exact video segment timestamps where that visual scene appears.

```
Input Video (.mp4)
       │
       ▼
[video/reader.py]  ────────► Stream frames & extract metadata
       │
       ▼
[video/scenes.py]  ────────► Detect scene transitions with HSV histograms
       │                      Apply segmentation rules (merge < 2s, split > 30s)
       │
       ▼
[models/clip_encoder.py] ──► Sample 3 frames per segment
       │                      Batch encode & average (mean pooling)
       │                      L2-normalize segment vectors
       │                      Unload OpenCLIP model
       ▼
[store/vectors.py] ────────► FAISS IndexFlatIP (Cosine similarity)
       ▲
       │ Query Text Vector (1×512)
       │
Text Query String ─────────► OpenCLIP text encoder
```

---

### Module 1: Video Reader (`src/videosdk/video/reader.py`)

#### Purpose
Extract video metadata (duration, FPS, resolution), seek to millisecond-accurate timestamps, stream frames without memory leaks, and compute file hashes for deduplication.

#### Important Code Sections & Why They Matter

1. **Chunked SHA-256 Hashing**:
   ```python
   def compute_file_hash(path: str | Path, chunk_size: int = 65536) -> str:
       hasher = hashlib.sha256()
       with file_path.open("rb") as f:
           while chunk := f.read(chunk_size):
               hasher.update(chunk)
       return hasher.hexdigest()
   ```
   *Why*: Reading files in 64 KB binary chunks ensures that 10 GB+ video files can be fingerprinted with virtually zero RAM footprint.

2. **Millisecond-Precise Seeking**:
   ```python
   def read_frame_at_time(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray:
       cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
       ret, frame = cap.read()
       ...
       return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   ```
   *Why*:
   - Seeking by frame number (`CAP_PROP_POS_FRAMES`) causes temporal drift on Variable Frame Rate (VFR) videos (common in phone recordings). `CAP_PROP_POS_MSEC` remains timestamp-accurate.
   - OpenCV decodes in **BGR** order by default. PIL and PyTorch/CLIP expect **RGB**. Converting immediately ensures downstream models never receive inverted color channels.

3. **Frame Generator / Streaming**:
   ```python
   def iter_frames_at_times(path: str | Path, times: Sequence[float]) -> Iterator[np.ndarray]:
       cap = cv2.VideoCapture(str(file_path))
       try:
           for t in times:
               yield read_frame_at_time(cap, t)
       finally:
           cap.release()
   ```
   *Why*: Uses Python's `yield` to stream one decoded uncompressed frame at a time. The capture is safely released in a `finally` block to prevent file descriptor leaks.

---

### Module 2: Scene Detection (`src/videosdk/video/scenes.py`)

#### Purpose
Divide raw continuous video into semantically meaningful shots/scenes using color histogram correlation, avoiding computationally expensive deep learning models for boundary detection.

#### Important Code Sections & Why They Matter

1. **HSV 2D Histograms**:
   ```python
   # Sample every 0.5s (2 frames/sec instead of 30 fps saves >90% processing time)
   hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
   # Hue: 50 bins, Saturation: 60 bins
   hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
   cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
   ```
   *Why*: The HSV color space separates chromatic content (Hue and Saturation) from illumination (Value). This makes scene detection resilient to slight lighting changes while detecting shot cuts.

2. **Histogram Correlation & Thresholding**:
   ```python
   correlation = float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL))
   if math.isnan(correlation) or correlation < threshold:
       boundaries.append(time_curr)
   ```
   *Why*: `HISTCMP_CORREL` gives `1.0` for identical frames and `-1.0` for completely dissimilar frames. A drop below `0.7` signals an abrupt scene transition.

3. **Segment Post-Processing Rules (`_apply_segment_rules`)**:
   - **Fallback Rule**: If `scene_count < duration / 30.0` (static presentations, podcasts, lectures with no scene changes), fallback to fixed 10-second windows.
   - **Merge Rule**: Any segment $< 2.0\text{s}$ is merged into an adjacent neighbor to eliminate sub-second flashes.
   - **Split Rule**: Any segment $> 30.0\text{s}$ is split into equal sub-windows of $\sim$10s to keep segment retrieval fine-grained.

---

### Module 3: Visual & Query Embedding (`src/videosdk/models/clip_encoder.py`)

#### Purpose
Transform visual frames and text search queries into the same 512-dimensional embedding space using OpenCLIP.

#### Important Code Sections & Why They Matter

1. **Lazy Loading & Offline Mode**:
   ```python
   os.environ.setdefault("HF_HUB_OFFLINE", "1")
   model, _, preprocess = open_clip.create_model_and_transforms(
       self.config.clip_model_name, # "ViT-B-32"
       pretrained=self.config.clip_pretrained, # "laion2b_s34b_b79k"
       device=self.device,
   )
   if self.device.type == "cuda":
       model.to(torch.float16) # Cut memory by 50%
   ```
   *Why*: Setting `HF_HUB_OFFLINE=1` prevents Hugging Face from hanging on network HEAD requests when running in air-gapped or offline environments.

2. **Multi-Frame Mean Pooling (3 Frames per Segment)**:
   ```python
   # Extract 3 frames across segment at 15%, 50%, and 85% timestamps
   offsets = [dur * (i + 1) / (k + 1) for i in range(k)]
   ...
   mean_vec = np.mean(seg_frames, axis=0)
   norm = np.linalg.norm(mean_vec)
   if norm > 1e-6:
       mean_vec = mean_vec / norm
   ```
   *Decision*: Why 3 frames instead of 1?
   - A single frame (e.g. center) misses scene evolution and action transitions.
   - Sampling 3 frames and taking the normalized mean represents the entire segment without excessive overhead.

3. **L2 Normalization**:
   ```python
   features = features / features.norm(dim=-1, keepdim=True)
   ```
   *Why*: Unit-normalizing every vector ensures that Inner Product search directly equals Cosine Similarity.

4. **Guaranteed Memory Unload**:
   ```python
   try:
       ...
   finally:
       encoder.unload() # Calls del, gc.collect(), torch.cuda.empty_cache()
   ```
   *Why*: Ensures that CLIP immediately vacates GPU memory once visual embeddings are computed, leaving the full 4 GB VRAM budget available for the next model.

---

### Module 4: FAISS Vector Store (`src/videosdk/store/vectors.py`)

#### Purpose
Store normalized vectors on disk and provide sub-millisecond similarity search.

#### Important Code Sections & Why They Matter

1. **Flat Inner Product Index (`IndexFlatIP`)**:
   ```python
   self.index = faiss.IndexFlatIP(self.dim)
   ```
   *Math Rationale*:
   $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
   When $\|\mathbf{u}\|_2 = 1.0$ and $\|\mathbf{v}\|_2 = 1.0$:
   $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \text{Inner Product}$$
   `IndexFlatIP` does exact exhaustive search using optimized AVX/SIMD instructions. For thousands of video segments, search takes less than 1 millisecond.

2. **Direct Row ID to Segment ID Mapping**:
   When segments are inserted into FAISS, index rows `0, 1, 2, ...` map directly to `segments[0], segments[1], ...`, eliminating the need for complex ID lookups in Phase 1.

---

### End-to-End Visual Search Script (`examples/visual_search_demo.py`)

A standalone demo demonstrating the entire Phase 1 pipeline:
```bash
python examples/visual_search_demo.py tests/data/sample_visual_search.mp4 "green forest"
```

---

## Phase 2 — Speech Search

### Overview & Pipeline

Phase 2 enables searching video content by what is **spoken**. The speech pipeline extracts audio, transcribes speech with word-level timestamps and VAD filtering, aligns transcripts with the visual scene segments, embeds the text using a lightweight sentence transformer on CPU, and indexes metadata in SQLite.

```
Input Video (.mp4)
       │
       ▼
[video/reader.py:extract_audio] ──► FFmpeg: 16 kHz Mono PCM WAV
       │
       ▼
[models/asr.py] ─────────────────► faster-whisper (CTranslate2 int8) + Silero VAD
       │                           Drop low-confidence text (avg_logprob < -1.0)
       │                           Unload Whisper model & clear VRAM
       ▼
[models/asr.py:align_transcript] ─► Align timestamped speech chunks to scene segments
       │
       ▼
[models/text_encoder.py] ────────► sentence-transformers (all-MiniLM-L6-v2) on CPU
       │                           Generate 384-dim L2-normalized vectors
       ▼
[store/vectors.py (text.faiss)] ─► FAISS IndexFlatIP (384-dim)
       │
       ▼
[store/metadata.py] ─────────────► SQLite (videos + segments tables with foreign keys)
```

---

### Module 1: Audio Extraction (`src/videosdk/video/reader.py:extract_audio`)

#### Purpose
Extract the audio stream from arbitrary video containers into standard 16 kHz mono 16-bit PCM WAV format expected by Whisper.

#### Key Code & Implementation
```python
def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
) -> str:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(src),
        "-vn",                 # Drop video stream
        "-acodec", "pcm_s16le",# Standard 16-bit linear PCM
        "-ar", str(sample_rate),# 16000 Hz
        "-ac", "1",            # Downmix to mono
        str(dst),
    ]
    subprocess.run(cmd, capture_output=True, text=True)
```
*Why*: 16 kHz mono PCM WAV is the universal format for Whisper. Extracting upfront saves model memory and allows Whisper to stream through the audio file without spawning internal FFmpeg processes.

---

### Module 2: Faster-Whisper ASR & Alignment (`src/videosdk/models/asr.py`)

#### Purpose
Transcribe spoken audio reliably while strictly respecting the 4 GB VRAM limit and suppressing hallucinated text during silent pauses.

#### Key Patterns & Code

1. **Voice Activity Detection (VAD) & Log-Probability Filtering**:
   ```python
   segments_gen, _info = self.model.transcribe(
       str(src),
       vad_filter=self.config.whisper_vad_filter,
       vad_parameters=dict(min_silence_duration_ms=500),
   )
   for seg in segments_gen:
       # Reject low-confidence hallucinations
       if seg.avg_logprob < self.config.whisper_min_confidence: # -1.0
           continue
       ...
   ```
   *Why*: Whisper models are known to hallucinate repetitive phrases during silent scenes or background noise. Turning on Silero VAD skips non-speech regions entirely, and the log-probability threshold eliminates low-confidence transcript artifacts.

2. **Temporal Alignment to Visual Segments**:
   ```python
   def align_transcript_to_segments(chunks, segments):
       # chunk overlaps [seg_start, seg_end] if chunk.start < seg_end and chunk.end > seg_start
       for idx, (seg_start, seg_end) in enumerate(segments):
           matching = [c.text for c in chunks if c.start < seg_end and c.end > seg_start]
           aligned[idx] = " ".join(matching).strip()
   ```
   *Why*: By aligning spoken text to the *same* temporal segments determined during visual scene detection, downstream multi-modal fusion (Phase 3) can cleanly correlate visual features and speech features for the exact same time window.

3. **Memory Isolation**:
   `WhisperTranscriber.unload()` calls `unload_model()` to clear CTranslate2/CUDA allocations before transcript embeddings or visual models execute.

---

### Module 3: Sentence-Transformers MiniLM (`src/videosdk/models/text_encoder.py`)

#### Purpose
Convert aligned transcript texts and search queries into 384-dimensional dense semantic vectors.

#### Key Decisions
- **Deliberate CPU Execution**: `SentenceTransformer("all-MiniLM-L6-v2", device="cpu")`. Because MiniLM is compact (~80 MB), CPU inference takes $<50\text{ ms}$ for typical video transcripts, preserving 100% of GPU memory for vision/ASR models.
- **Normalization**: Vectors are unit-normalized (`normalize_embeddings=True`) so inner product search corresponds to cosine similarity.

---

### Module 4: SQLite Metadata Store (`src/videosdk/store/metadata.py`)

#### Purpose
Provide ACID relational persistence for video catalog records, segment time boundaries, transcripts, and row pointers into `visual.faiss` and `text.faiss`.

#### Schema & Key Implementation
```sql
CREATE TABLE videos (
    video_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL UNIQUE,
    file_hash   TEXT NOT NULL,
    duration    REAL NOT NULL,
    fps         REAL NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    added_at    TEXT NOT NULL
);

CREATE TABLE segments (
    segment_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        INTEGER NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    start_time      REAL NOT NULL,
    end_time        REAL NOT NULL,
    transcript_text TEXT DEFAULT '',
    visual_index_id INTEGER NOT NULL,
    text_index_id   INTEGER DEFAULT -1,
    UNIQUE(video_id, start_time)
);
```
- Foreign key support enabled via `PRAGMA foreign_keys = ON;`.
- Deleting a video via `store.remove_video(path)` automatically cascades to remove its segment entries.
- Indexed lookups by `visual_index_id` and `text_index_id` facilitate $O(1)$ result construction during search fusion.

---

### Verification & Test Results

All 38 unit and integration tests across the project pass with 0 errors and 0 warnings:

| Test File | Focus | Results |
|---|---|---|
| `tests/test_reader.py` | Metadata, seeking, streaming, SHA-256 hash, audio extraction | 11 / 11 Passed |
| `tests/test_scenes.py` | Histogram boundary detection, merge/split rules, static video fallback | 6 / 6 Passed |
| `tests/test_asr.py` | TranscriptChunk dataclass, temporal alignment, VAD confidence filtering, unload | 6 / 6 Passed |
| `tests/test_text_encoder.py` | MiniLM CPU encoding, empty text handling, query vector shape, lifecycle helper | 5 / 5 Passed |
| `tests/test_store.py` | FAISS vector store + SQLite relational metadata CRUD, cascade delete, dedup | 6 / 6 Passed |
| `tests/test_api.py` | Base types (`SearchResult`, `VideoMeta`), configuration defaults, error types | 4 / 4 Passed |

Code style and formatting:
- `ruff check src/ tests/`: **100% Clean (0 errors)**
- `ruff format --check src/ tests/`: **100% Formatted**

---

## Phase 3 — Fusion & Ranking

### Overview & Pipeline

Phase 3 merges the multi-modal retrieval streams from Phase 1 (visual content) and Phase 2 (spoken audio) into a single, cohesive ranking.

```
                              Query String
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
         [Visual Search Mode]              [Speech Search Mode]
          OpenCLIP Text Encoder             MiniLM SentenceTransformer
          (Load -> Encode -> Unload)        (Load -> Encode -> Unload)
                  │                                 │
                  ▼                                 ▼
          visual.faiss (512-dim)            text.faiss (384-dim)
          Top-K Visual Candidates           Top-K Speech Candidates
                  │                                 │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                   [Reciprocal Rank Fusion (RRF)]
                   Formula: SUM( 1 / (60 + rank_i) )
                                   │
                                   ▼
                   [SQLite Metadata Resolution]
                   Fetch video path, time bounds, transcript text
                   Format timestamps (MM:SS / H:MM:SS)
                   Attribute source: 'visual' | 'speech' | 'both'
                                   │
                                   ▼
                       List[SearchResult] Objects
```

---

### Module 1: Reciprocal Rank Fusion (`src/videosdk/search/fusion.py`)

#### Purpose
Combine ranked candidate lists from separate retrieval channels into a unified ranking without score calibration.

#### Why RRF Over Weighted Score Averaging
1. **Incompatible Score Scales**:
   CLIP cosine similarity and MiniLM cosine similarity produce completely different value distributions. A score of `0.28` in CLIP might represent a very strong match for an abstract visual concept, while `0.28` in MiniLM might be a weak match. Directly summing or averaging them introduces strong bias towards whichever model outputs higher raw numbers.
2. **Rank-Based Invariance**:
   RRF uses only the *position* (1st, 2nd, 3rd...) of candidates in each list:
   $$\text{RRF\_score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
   where $k = 60$ is the standard smoothing constant (Cormack, Clarke, and Buettcher, 2009) and $r_m(d)$ is the 1-based rank position in modality $m$.
3. **Synergy Boost**:
   Candidates appearing in both modalities receive a significant reciprocal boost, naturally surfacing scenes where both visual cues and spoken dialogue match the query.

#### Key Implementation
```python
def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    if not ranked_lists:
        return []

    rrf_scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (item_id, _original_score) in enumerate(ranked_list, start=1):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (k + rank))

    return sorted(rrf_scores.items(), key=lambda entry: entry[1], reverse=True)
```

---

### Module 2: Unified Search Coordinator (`src/videosdk/search/searcher.py`)

#### Purpose
Provide a single search entry point supporting visual-only, speech-only, and hybrid multi-modal search while enforcing strict memory isolation.

#### Key Features & Implementation Details

1. **Strict Sequential Model Execution**:
   - `_visual_search()` loads `CLIPEncoder`, encodes the query into a 512-dim vector, immediately executes `encoder.unload()`, and queries `visual_store`.
   - `_speech_search()` loads `TextEncoder` on CPU, encodes the query into a 384-dim vector, immediately executes `encoder.unload()`, and queries `text_store`.
   - The two encoders **never** coexist in memory, preserving the 4 GB VRAM ceiling.

2. **Source Attribution**:
   When searching with `mode="both"`, the coordinator determines how each segment was discovered:
   ```python
   in_visual = seg_id in visual_ids
   in_speech = seg_id in speech_ids
   if in_visual and in_speech:
       source = "both"
   elif in_visual:
       source = "visual"
   else:
       source = "speech"
   ```

3. **Human-Readable Timestamp Formatting**:
   ```python
   def format_timestamp(seconds: float) -> str:
       total_seconds = max(0, round(seconds))
       hours, remainder = divmod(total_seconds, 3600)
       minutes, secs = divmod(remainder, 60)
       if hours > 0:
           return f"{hours}:{minutes:02d}:{secs:02d}"
       return f"{minutes:02d}:{secs:02d}"
   ```
   Converts seconds into `"MM:SS"` or `"H:MM:SS"` (e.g. `75.0` $\rightarrow$ `"01:15"`, `3665.0` $\rightarrow$ `"1:01:05"`).

4. **Result Packaging**:
   Hydrates SQLite metadata into public `SearchResult` dataclasses containing video path, start/end seconds, formatted timestamp, score, transcript text, and source attribution.

---

### Verification & Test Results

All 8 tests in `tests/test_fusion.py` pass:
- `test_rrf_basic`: Verifies mathematical scoring and overlapping item boost.
- `test_rrf_single_list`: Handles single modality without degradation.
- `test_rrf_empty`: Safely returns empty list on empty inputs.
- `test_rrf_no_overlap`: Merges non-overlapping candidate lists cleanly.
- `test_rrf_custom_k`: Validates configurable smoothing constant $k$.
- `test_format_timestamp`: Validates timestamps across zero, seconds, minutes, and hours.
- `test_searcher_validation_and_empty_query`: Rejects invalid modes and whitespace queries.
- `test_searcher_modes_and_fusion`: Validates visual, speech, and hybrid both modes with source attribution.

---

## Phase 4 — SDK Public API & CLI *(Upcoming)*
- [ ] `VideoIndex` high-level class (`index_video`, `search`, `save`, `load`)
- [ ] Typer CLI (`videosdk index`, `videosdk search`, `videosdk clip`)

## Phase 4.5 — Advanced Multimodal Intelligence & Creator Tools *(Upcoming)*
- [ ] **Reverse Visual Search (Image-to-Video)**: Query visual FAISS index using image references (`index.search(image=...)`) via OpenCLIP `encode_image`.
- [ ] **Automated Supercut / Highlight Reel Generator**: Stitch candidate clips into a continuous reel (`index.create_supercut(...)`) with temporal merging and FFmpeg concat/xfade.
- [ ] **Video Editor & Subtitle Exporters**: Direct export of search results to `.srt`, `.vtt`, `.edl` (Premiere / DaVinci Resolve), `.fcpxml` (Final Cut Pro), and `.csv`.
- [ ] **Smart Video Chapters & Timeline Heatmaps**: Unsupervised topic clustering into structured chapters and dense query-relevance scrub bar heatmaps.
- [ ] **Non-Speech Audio Event Search**: Acoustic scene retrieval (clapping, gunshots, laughter, rain) using lightweight CLAP embeddings under sequential VRAM lifecycle.
- [ ] **Embedded Zero-Node Local Web UI**: Built-in browser app (`videosdk ui`) with deep-linked video playback, interactive timeline scrubbers, and clip export.

## Phase 5 — Testing & Evaluation Benchmarks *(Upcoming)*
- [ ] Evaluation benchmark with synthetic & real video ground truths
- [ ] Recall@1, Recall@5, and latency benchmarks

## Phase 6 — Documentation & Distribution *(Upcoming)*
- [ ] Documentation, quickstart guide, PyPI packaging verification
