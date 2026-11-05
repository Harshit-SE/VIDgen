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
4. [Phase 2 — Speech Search](#phase-2--speech-search-upcoming) *(Upcoming)*
5. [Phase 3 — Fusion & Ranking](#phase-3--fusion--ranking-upcoming) *(Upcoming)*
6. [Phase 4 — SDK Public API & CLI](#phase-4--sdk-public-api--cli-upcoming) *(Upcoming)*
7. [Phase 5 — Testing & Evaluation Benchmarks](#phase-5--testing--evaluation-benchmarks-upcoming) *(Upcoming)*
8. [Phase 6 — Documentation & Distribution](#phase-6--documentation--distribution-upcoming) *(Upcoming)*

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

#### Demo Output
```text
--- VideoSDK Phase 1 Visual Search Prototype ---
Target video : sample_visual_search.mp4
Search query : 'green forest'
Compute device: cpu

[1/4] Detecting scene boundaries...
      Detected 3 segments.

[2/4] Embedding visual content (OpenCLIP)...
      Generated embedding matrix: (3, 512)

[3/4] Populating FAISS vector index...
      Indexed 3 segment vectors.

[4/4] Encoding query and searching top-5 results...

--- Top Search Results ---
#1 | Score: 0.2319 | Timestamp: 00:03 (3.0s - 6.0s)
#2 | Score: 0.1404 | Timestamp: 00:00 (0.0s - 3.0s)
#3 | Score: 0.1271 | Timestamp: 00:06 (6.0s - 9.0s)
```

---

### Verification & Test Results

All 25 automated tests across the project pass:

| Test File | Focus | Results |
|---|---|---|
| `tests/test_reader.py` | Metadata extraction, seeking, streaming generator, SHA-256 chunking, missing/corrupted video error handling | 7 / 7 Passed |
| `tests/test_scenes.py` | Histogram boundary detection, segment merge (<2s), segment split (>30s), fallback to 10s windows, synthetic video cuts | 6 / 6 Passed |
| `tests/test_store.py` | FAISS index creation, vector insertion, inner product search, dimension mismatch validation, persistence to disk | 4 / 4 Passed |
| `tests/test_clip_encoder.py` | Text query embedding, frame batch embedding, L2 normalization, segment embedding pipeline, empty list handling | 4 / 4 Passed |
| `tests/test_api.py` | Base types (`SearchResult`, `VideoMeta`), configuration defaults, custom exceptions | 4 / 4 Passed |

Code style and type checks:
- `ruff check src tests examples`: **100% Clean (0 errors)**
- `mypy src`: **Passing with strict typed definitions**

---

## Phase 2 — Speech Search *(Upcoming)*
*Goal*: Search video by spoken audio.
- [ ] FFmpeg audio extraction to 16 kHz mono WAV
- [ ] `faster-whisper` transcription with Voice Activity Detection (VAD)
- [ ] Transcript chunk alignment to Phase 1 video segments
- [ ] `all-MiniLM-L6-v2` sentence transformer embeddings for transcripts
- [ ] Speech vector index in FAISS and SQLite metadata storage
- [ ] Sequential model execution memory validation

## Phase 3 — Fusion & Ranking *(Upcoming)*
*Goal*: Merge visual and speech candidate lists into a unified ranked result set.
- [ ] Reciprocal Rank Fusion (RRF with $k=60$)
- [ ] Search modes: `"visual"`, `"speech"`, `"both"`
- [ ] End-to-end `SearchResult` dataclass with timestamp formatting and clipping

## Phase 4 — SDK Public API & CLI *(Upcoming)*
- [ ] `VideoIndex` high-level class (`index_video`, `search`, `save`, `load`)
- [ ] Typer CLI (`videosdk index`, `videosdk search`, `videosdk clip`)

## Phase 5 — Testing & Evaluation Benchmarks *(Upcoming)*
- [ ] Evaluation benchmark with synthetic & real video ground truths
- [ ] Recall@1, Recall@5, and latency benchmarks

## Phase 6 — Documentation & Distribution *(Upcoming)*
- [ ] Documentation, quickstart guide, PyPI packaging verification

