# VideoSDK (working name): Search Inside Videos

> **For the coding agent (Codex / Antigravity):** This file is the source of truth for what to build.
> Read this file fully, then read `TRACKER.md`. Work one phase at a time. Update `TRACKER.md` after every task.
> Do not invent features outside the scope in section 4.

---

## 1. What is it?

A Python library (SDK) that lets developers search inside videos by what is **said** and what is **seen**.
It runs locally on a small GPU (4 GB) and also on CPU.

- Input: a video file.
- Output: a searchable index, and for any text query, ranked timestamps plus short clips.
- Runs offline. No cloud APIs. No API keys.

---

## 2. What problem does it solve?

- Videos are hard to search. People scrub the timeline by hand to find one moment.
- Cloud video-search services cost money and need you to upload private videos.
- Most open-source examples are tutorials, not reusable libraries.
- Most tools need a big GPU.

**Answer:** a small, clean, offline library that works on modest hardware and is easy to plug into any app.

---

## 3. Who benefits?

- **Students and teachers:** find the exact part of a long lecture where a topic is explained.
- **Developers:** add video search to their app in a few lines, without building the pipeline.
- **Content creators and editors:** find a moment in hours of footage by describing it.
- **Small teams and researchers:** search meeting recordings and interviews privately.
- **People with weak hardware:** designed for a 4 GB GPU or CPU.

> Note: these users are assumed, not validated. See Risk 10 in section 11.

---

## 4. Scope

### In scope (version 1)

- Add a video, build an index, search by text.
- Search by visual content and by speech.
- Combine both into one ranked result list.
- Return timestamps, a score, the matched transcript text, and a short clip.
- Python API and a command line tool (CLI).
- Fully offline.
- Runs on a 4 GB GPU, with a CPU fallback.

### Out of scope

- Video generation or video editing.
- Web app, server, or hosted service.
- Live or streaming video.
- LLM-written answers.
- Image or audio search as input (text queries only).
- Multi-user features, accounts, or a database server.

---

## 5. Public API (this is the contract, do not change without asking)

```python
from videosdk import VideoIndex

# Create or open an index stored in a folder
index = VideoIndex("my_index", device="auto")   # device: "auto" | "cuda" | "cpu"

# Add videos (can be called many times)
index.add("lecture.mp4")

# Search
results = index.search("where is backpropagation explained?", top_k=5, mode="both")
# mode: "visual" | "speech" | "both"

best = results[0]
best.video        # path of the source video
best.start        # seconds (float)
best.end          # seconds (float)
best.timestamp    # "12:45" (string)
best.score        # float, higher is better
best.text         # matched transcript text (may be empty)
best.source       # "visual" | "speech" | "both"

best.clip("clip.mp4")   # saves a short clip of the match

# Utilities
index.list_videos()
index.remove("lecture.mp4")
```

CLI:

```
videosdk add my_index lecture.mp4
videosdk search my_index "backpropagation" --top-k 5 --mode both
videosdk list my_index
videosdk remove my_index lecture.mp4
```

Rules for the API:
- All public functions have type hints and docstrings.
- Errors are clear and tell the user what to do (example: "Video file not found: X").
- Return typed objects (dataclass or pydantic), not raw dicts.

---

## 6. How it works inside

### Indexing pipeline (when `add()` is called)

1. **Validate** the file and read metadata (duration, fps, resolution) with OpenCV / ffprobe.
2. **Split into segments** with scene detection (OpenCV). Rules:
   - Minimum segment length: 2 seconds. Merge shorter ones into neighbors.
   - Maximum segment length: 30 seconds. Split longer ones into equal windows.
   - Fallback: if scene detection finds very few scenes (example: a static slide lecture), use fixed 10-second windows.
3. **Visual embeddings:** for each segment, take up to 3 frames (start, middle, end). Embed with CLIP. Average or keep the best per segment (decide in Phase 1 and log it in the tracker).
4. **Free GPU memory** (unload the CLIP model).
5. **Speech to text:** run Whisper on the audio (with voice activity detection to skip silence). Get text with timestamps.
6. **Free GPU memory** (unload the Whisper model).
7. **Transcript chunks:** group transcript text into chunks aligned to the same segments. Embed each chunk with a small text embedding model.
8. **Store** everything on disk (see section 8).

Models are loaded one at a time, never together. This is the key rule for the 4 GB limit.

### Search pipeline (when `search()` is called)

1. Embed the query with the CLIP text encoder, then search the visual index.
2. Embed the query with the text embedding model, then search the transcript index.
3. Combine both ranked lists with **Reciprocal Rank Fusion (RRF)**. It needs no score tuning.
4. Return the top K segments with timestamps, scores, and text.

---

## 7. Tech stack

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Standard for ML tools |
| Video reading and scene detection | OpenCV (`opencv-python-headless`) | Core requirement of the project |
| Video cutting and audio extraction | FFmpeg (system install) | Fast, reliable, standard |
| Visual embeddings | CLIP ViT-B/32 via `open_clip_torch` (or Hugging Face `transformers`) | Small, fits 4 GB |
| Speech to text | `faster-whisper` with `base` or `small` model, int8 or fp16 | Uses less memory and runs faster than the original Whisper |
| Transcript embeddings | `sentence-transformers` with `all-MiniLM-L6-v2` | Very small, runs on CPU too |
| Vector search | FAISS (`faiss-cpu`) | No GPU memory used, simple, fast enough |
| Metadata storage | SQLite (Python built-in `sqlite3`) | No server, one file |
| Deep learning runtime | PyTorch | Needed by CLIP and MiniLM |
| CLI | Typer | Simple and typed |
| Data types | dataclasses or pydantic | Clear result objects |
| Packaging | `pyproject.toml` (hatchling or setuptools) | Modern standard, needed for PyPI |
| Testing | pytest | Standard |
| Code quality | ruff (lint and format), mypy (types) | Shows professionalism |
| CI | GitHub Actions | Runs tests on every push |
| Docs | README, `docs/` folder, one demo notebook | Recruiter-visible |

Rules for dependencies:
- Pin versions in `pyproject.toml` (use ranges that are tested, and record the exact tested versions in the tracker).
- Core install must be simple: `pip install videosdk`. Heavy parts (GPU build of PyTorch) are documented, not forced.
- Do not add a new dependency without writing the reason in the tracker's decision log.

---

## 8. Repo structure

```
videosdk/
├── pyproject.toml
├── README.md
├── PROJECT_OVERVIEW.md
├── TRACKER.md
├── LICENSE
├── src/videosdk/
│   ├── __init__.py          # exports VideoIndex
│   ├── index.py             # VideoIndex class (public API)
│   ├── types.py             # SearchResult and other typed objects
│   ├── config.py            # settings, model names, limits
│   ├── errors.py            # custom errors with clear messages
│   ├── video/
│   │   ├── reader.py        # metadata, frame reading
│   │   ├── scenes.py        # scene detection + segment rules
│   │   └── clipper.py       # cut a clip with FFmpeg
│   ├── models/
│   │   ├── clip_encoder.py  # load, embed, unload
│   │   ├── asr.py           # faster-whisper wrapper
│   │   ├── text_encoder.py  # MiniLM wrapper
│   │   └── memory.py        # helpers to free GPU memory
│   ├── store/
│   │   ├── vectors.py       # FAISS read/write
│   │   └── metadata.py      # SQLite read/write
│   ├── search/
│   │   ├── fusion.py        # RRF
│   │   └── searcher.py      # search logic
│   └── cli.py               # Typer CLI
├── tests/
│   ├── test_scenes.py
│   ├── test_fusion.py
│   ├── test_store.py
│   ├── test_api.py
│   └── data/                # tiny test videos (a few seconds each)
├── benchmarks/
│   ├── run_benchmark.py
│   ├── eval_set/            # videos list, queries, labeled answers
│   └── results/             # saved result tables
├── examples/
│   └── demo.ipynb
└── .github/workflows/ci.yml
```

### On-disk index format (inside the index folder)

- `meta.sqlite`: videos table and segments table (video id, start, end, transcript text).
- `visual.faiss`: visual vectors.
- `text.faiss`: transcript vectors.
- `index.json`: `schema_version`, model names used, creation date.

Always write `schema_version`. If the version does not match, show a clear error instead of crashing.

---

## 9. Hardware limits and rules

Target machine: NVIDIA RTX 3050, 4 GB memory.

- Models load one at a time. Always unload and call `torch.cuda.empty_cache()` after each stage.
- Use fp16 on GPU where supported.
- Process in small batches. Make batch size a setting. Catch out-of-memory errors, then retry with a smaller batch automatically.
- Whisper: default `base`, allow `small`. Never default to `medium` or `large`.
- Long videos: read frames as a stream. Do not load the whole video in memory.
- CPU fallback: if no GPU is found, or GPU runs out of memory, continue on CPU with a clear warning.
- Report peak memory in benchmarks.

---

## 10. How success is measured

Only claim what is measured.

- **Speed:** processing time per 1 hour of video, per stage (scenes, visual, speech, storing).
- **Memory:** peak GPU memory and peak RAM.
- **Search speed:** time per query.
- **Accuracy:** an eval set made by the owner (not by the agent).
  - 5 to 10 videos (use Creative Commons or self-made videos).
  - 30 to 50 queries, each labeled with the correct time range.
  - Metrics: Recall@1, Recall@5, and MRR.
  - Compare: visual only vs speech only vs both.
- **CPU vs GPU:** same test on both, report honestly.

Results go into the README as a table and into `benchmarks/results/`.

---

## 11. Risks and how to overcome each one

| # | Risk | Why it matters | How to overcome |
|---|---|---|---|
| 1 | Looks like a tutorial project (CLIP + Whisper glued together) | Recruiters see many of these | Win on quality and proof: real benchmarks, clean API, tests, CI, PyPI release, honest write-up. Add one small original idea (for example: fallback segmentation for static lectures, or the RRF fusion comparison study). |
| 2 | GPU runs out of memory (4 GB) | Crashes make the SDK feel broken | Load one model at a time, unload after use, fp16, small batches, auto-retry with smaller batch, CPU fallback. Test on the real RTX 3050 in Phase 1 and 2, not at the end. |
| 3 | Slow on long videos | Whisper on a 2-hour video is slow | Stream frames, process in chunks, allow smaller model, show a progress bar, cache results so a video is never processed twice. Report time per hour of video honestly. |
| 4 | Weak or fake accuracy claims | Interviewers will question numbers | Owner builds the eval set by hand. Fixed queries, saved results, reproducible script. If results are poor, report them and explain why. |
| 5 | "Works on CPU" claim is untested | Easy to get caught | Add a CPU test run in CI (small video) and in benchmarks. Do not write the claim in the README until it passes. |
| 6 | Scope creep | Project never finishes | Section 4 is the limit. Anything new goes to a "Later" list in the tracker. Finish phases in order. |
| 7 | Package name already taken on PyPI | Cannot publish | Check PyPI in Phase 0. Pick the final name, then rename once. Keep the name in one config place. |
| 8 | Install problems (CUDA, PyTorch, FAISS, FFmpeg) | Users give up on the first try | Document install steps in the README. Add a `videosdk doctor` command that checks Python, FFmpeg, GPU, and model files and prints what is missing. |
| 9 | Scene detection fails on some videos (static slides, constant motion) | Bad segments mean bad search | Segment length limits (2s min, 30s max) and fixed-window fallback. Test on at least one lecture video and one vlog-style video. |
| 10 | Nobody wants it (users not validated) | A project without users is weak | Before Phase 4, show a demo to 3 to 5 people (classmates, developers) and ask if they would use it. Write what they said in the tracker. Adjust the README message to what they care about. |
| 11 | AI agent writes wrong or hallucinated code | Silent bugs and fake API calls | Work in small phases. Every phase needs passing tests before moving on. Pin versions. Owner reads and understands every module (the owner must be able to explain it in an interview). |
| 12 | Whisper makes up text on silent parts | Wrong search hits | Turn on voice activity detection. Drop very low-confidence segments. Test on a video with long silences. |
| 13 | License and copyright problems | Legal risk, cannot share demo | Use Creative Commons or self-made videos for tests and demos. Check the license of each model and list them in the README. |
| 14 | Index gets large | Disk fills up | Store only vectors and small metadata, not frames. Use float32 vectors with a note on size. Report index size per hour of video. |
| 15 | Index format changes later and breaks old indexes | Bad user experience | `schema_version` in `index.json`, clear error on mismatch. |
| 16 | Deadline slips | Unfinished project is worth little | Phases 0 to 4 give a working SDK. Phases 5 and 6 add proof and polish. If time is short, cut polish, never cut tests or benchmarks. |

---

## 12. Rules for the coding agent

- Follow this file and `TRACKER.md`. Do not change the public API (section 5) without asking the owner.
- Work on one phase at a time. Stop at the end of each phase and wait for the owner to review.
- After every task, update `TRACKER.md` (checkbox, date, notes).
- Write tests for every module. A phase is not done until its tests pass.
- Keep functions small and typed. Add docstrings to public code.
- Never claim a number (speed, accuracy, memory) that was not measured. Put measured numbers in the tracker's benchmark log.
- Do not add dependencies without logging the reason in the tracker.
- Do not create the eval set labels. The owner does that.
- If something is unclear, ask. Do not guess.
- Explain each new module in a few lines in the tracker's session log so the owner can learn it.

---

## 13. Definition of done (version 1)

- [ ] `pip install` works on a clean machine.
- [ ] Public API in section 5 works as written.
- [ ] CLI works.
- [ ] Runs on RTX 3050 (4 GB) without crashing on a 1-hour video.
- [ ] CPU fallback works and is tested.
- [ ] Tests pass in CI.
- [ ] Benchmarks are in the README with an honest table.
- [ ] README has a demo GIF, install steps, quick start, and limits.
- [ ] Published on PyPI.
- [ ] Short write-up of design choices and what failed.

---

## 14. Resume angle

- One line: "Built an offline video search SDK (OpenCV, CLIP, Whisper, FAISS) that runs on a 4 GB GPU; published on PyPI with benchmarks."
- Only use numbers from your own benchmarks.
- Be ready to explain: why RRF, why one model at a time, why segments, what failed.
