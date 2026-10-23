# TRACKER: VideoSDK build tracker

> **For the coding agent:** Update this file after every task. Tick the box, add the date, add short notes.
> Work one phase at a time. Stop at the end of each phase and wait for the owner to review.
> Source of truth for scope and API: `PROJECT_OVERVIEW.md`.

**Status legend:** `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` blocked

---

## Overall progress

| Phase | Name | Status | Started | Finished |
|---|---|---|---|---|
| 0 | Setup and decisions | [ ] | | |
| 1 | Visual search prototype | [ ] | | |
| 2 | Speech search | [ ] | | |
| 3 | Combine and rank | [ ] | | |
| 4 | Turn it into an SDK | [ ] | | |
| 5 | Tests, eval set and benchmarks | [ ] | | |
| 6 | Docs, polish and release | [ ] | | |

**Owner-only tasks** (the agent must not do these): label the eval set, show the demo to real users, final review of every phase, publish to PyPI.

---

## Phase 0: Setup and decisions

**Goal:** a clean repo where everything installs and runs.

- [ ] Check the package name on PyPI. Pick the final name. Keep it in one config place. (Owner)
- [x] Create repo with the structure in `PROJECT_OVERVIEW.md` section 8
- [ ] Create `pyproject.toml` with ruff, mypy, pytest config
- [ ] Add LICENSE (owner chooses)
- [ ] Check FFmpeg, Python version, CUDA, and GPU are working on the owner's machine
- [ ] Add a tiny GPU memory check script (prints total and free memory)
- [ ] Set up GitHub Actions: lint + tests on every push
- [ ] Choose and download 2 to 3 test videos (Creative Commons or self-made, short ones for tests, one 10-minute one for experiments)

**Done when:** `pip install -e .` works, `pytest` runs (even with one dummy test), CI is green.

**Risks covered:** 7 (name), 8 (install), 13 (licenses).

---

## Phase 1: Visual search prototype

**Goal:** type a text query, get the right moment in a video by what is seen.

- [ ] `video/reader.py`: read metadata and frames in a stream
- [ ] `video/scenes.py`: scene detection with OpenCV
- [ ] Segment rules: min 2s, max 30s, fixed 10s fallback for static videos
- [ ] `models/clip_encoder.py`: load, embed frames and text, unload
- [ ] `models/memory.py`: free GPU memory helper, out-of-memory retry with smaller batch
- [ ] `store/vectors.py`: save and load FAISS index
- [ ] Plain script: video in, query in, top 5 timestamps out
- [ ] Decide: 1 frame vs 3 frames per segment, average vs best (log in decision log)
- [ ] Tests for scenes and segment rules (`tests/test_scenes.py`)
- [ ] Measure peak GPU memory on the RTX 3050 and log it

**Done when:** the script works on the 10-minute test video, peak GPU memory is logged, scene tests pass.

**Risks covered:** 2 (GPU memory), 9 (bad scene detection).

---

## Phase 2: Speech search

**Goal:** search by what is said.

- [ ] Audio extraction with FFmpeg
- [ ] `models/asr.py`: faster-whisper wrapper with voice activity detection, unload after use
- [ ] Drop very low-confidence text
- [ ] Align transcript text to the same segments as Phase 1
- [ ] `models/text_encoder.py`: MiniLM wrapper
- [ ] Separate FAISS index for transcript vectors
- [ ] `store/metadata.py`: SQLite tables for videos and segments (with transcript text)
- [ ] Models run one after another (never together). Verify with a memory log.
- [ ] Test on a video with long silences
- [ ] Test on a video with no speech at all (must not crash)
- [ ] Measure Whisper time per hour of video and peak memory, then log it

**Done when:** speech-only search works, silent and no-speech videos do not break it, memory stays under 4 GB.

**Risks covered:** 2, 3 (slow), 12 (made-up text).

---

## Phase 3: Combine and rank

**Goal:** one ranked list from both search types.

- [ ] `search/fusion.py`: Reciprocal Rank Fusion
- [ ] `search/searcher.py`: visual, speech, and both modes
- [ ] Result object with video, start, end, timestamp, score, text, source
- [ ] Tests for fusion with small hand-made rankings (`tests/test_fusion.py`)
- [ ] Quick manual check: 10 queries, compare visual only vs speech only vs both

**Done when:** all three modes work and fusion tests pass.

**Risks covered:** 1 (adds an original comparison to the write-up).

---

## Phase 4: Turn it into an SDK

**Goal:** the public API in `PROJECT_OVERVIEW.md` section 5 works exactly as written.

- [ ] `types.py` and `errors.py` with clear error messages
- [ ] `index.py`: `VideoIndex` with `add`, `search`, `list_videos`, `remove`
- [ ] `index.json` with `schema_version`, model names, date
- [ ] Clear error on schema mismatch
- [ ] Cache: never process the same video twice (use a file hash)
- [ ] `video/clipper.py`: `result.clip()` using FFmpeg
- [ ] Device handling: `"auto" | "cuda" | "cpu"` and CPU fallback with warning
- [ ] `cli.py` with Typer: add, search, list, remove
- [ ] `videosdk doctor` command
- [ ] Progress bar while indexing
- [ ] Type hints and docstrings on all public code, mypy passes
- [ ] **Owner:** show a working demo to 3 to 5 people and write their feedback below

**User feedback log (owner fills):**

| Person | Would they use it? | What they said |
|---|---|---|
| | | |

**Done when:** the example code in section 5 runs without edits, CLI works, mypy and ruff pass.

**Risks covered:** 5, 8, 10, 15.

---

## Phase 5: Tests, eval set and benchmarks

**Goal:** proof. Real numbers.

- [ ] **Owner:** build the eval set (5 to 10 videos, 30 to 50 labeled queries) in `benchmarks/eval_set/`
- [ ] `benchmarks/run_benchmark.py`: one command, reproducible
- [ ] Metrics: Recall@1, Recall@5, MRR
- [ ] Compare: visual only, speech only, both
- [ ] Speed: time per hour of video per stage
- [ ] Memory: peak GPU and RAM
- [ ] Query speed: time per search
- [ ] Index size per hour of video
- [ ] Run everything on GPU and on CPU
- [ ] Test a 1-hour video end to end on the RTX 3050 without crashing
- [ ] Save results in `benchmarks/results/`
- [ ] Add a CPU test with a tiny video to CI
- [ ] Test coverage on core modules

**Done when:** one command reproduces all numbers and results are saved.

**Risks covered:** 3, 4, 5, 14.

---

## Phase 6: Docs, polish and release

**Goal:** something a stranger can install and understand in 30 seconds.

- [ ] README: one-line pitch, demo GIF, install, quick start, API, CLI, limits, benchmark table
- [ ] Say clearly what does not work well (honest limits)
- [ ] Demo notebook in `examples/demo.ipynb`
- [ ] List model licenses in the README
- [ ] Short write-up: design choices, what failed, what you would do next
- [ ] Final check that every claim in the README matches `benchmarks/results/`
- [ ] **Owner:** build and publish to PyPI (test on TestPyPI first)
- [ ] **Owner:** test `pip install` on a clean machine or fresh environment
- [ ] Add resume bullet using only measured numbers

**Done when:** the Definition of Done in `PROJECT_OVERVIEW.md` section 13 is fully ticked.

**Risks covered:** 1, 13, 16.

---

## Benchmark log (fill with measured numbers only)

| Date | Phase | Test | Hardware | Result |
|---|---|---|---|---|
| | | Peak GPU memory, CLIP stage | RTX 3050 4GB | |
| | | Peak GPU memory, Whisper stage | RTX 3050 4GB | |
| | | Scene detection time per hour of video | | |
| | | Whisper time per hour of video | RTX 3050 4GB | |
| | | Search time per query | | |
| | | Recall@1 / Recall@5 / MRR, visual only | | |
| | | Recall@1 / Recall@5 / MRR, speech only | | |
| | | Recall@1 / Recall@5 / MRR, both | | |
| | | CPU vs GPU indexing time | | |
| | | Index size per hour of video | | |

---

## Decision log

| Date | Decision | Reason | Alternatives rejected |
|---|---|---|---|
| | Package name: | | |
| | Frames per segment: | | |
| | Whisper model size default: | | |
| | Batch size default: | | |
| | | | |

**Tested dependency versions (record exact versions that work):**

- Python:
- PyTorch:
- opencv-python-headless:
- open_clip_torch:
- faster-whisper:
- sentence-transformers:
- faiss-cpu:
- typer:

---

## Risk checklist

Full table is in `PROJECT_OVERVIEW.md` section 11. Tick each one when its fix is in place and tested.

- [ ] 1. Looks like a tutorial project: benchmarks, tests, CI, release, write-up
- [ ] 2. GPU out of memory: one model at a time, fp16, batch retry, CPU fallback
- [ ] 3. Slow on long videos: streaming, chunking, caching, progress bar
- [ ] 4. Weak or fake accuracy claims: owner-made eval set, reproducible script
- [ ] 5. CPU claim untested: CPU test in CI and benchmarks
- [ ] 6. Scope creep: "Later" list is used, scope unchanged
- [ ] 7. PyPI name clash: name checked in Phase 0
- [ ] 8. Install problems: README steps and `videosdk doctor`
- [ ] 9. Bad scene detection: segment limits, fallback, tested on 2 video types
- [ ] 10. Nobody wants it: feedback from 3 to 5 people logged
- [ ] 11. Agent writes wrong code: tests per phase, pinned versions, owner reviews
- [ ] 12. Made-up text on silence: voice activity detection, tested on silent video
- [ ] 13. License problems: CC videos only, model licenses listed
- [ ] 14. Large index: size measured and reported
- [ ] 15. Format changes: `schema_version` check in place
- [ ] 16. Deadline slips: Phases 0 to 4 protected, polish cut first

---

## "Later" list (ideas that are NOT in version 1)

- (empty)

## Blockers

- (none)

## Session log (agent: add a short entry after each session)

| Date | What was done | Files changed | What the owner should learn from it |
|---|---|---|---|
| 2026-10-24 | Created the repository scaffold and copied the project overview and tracker into the root. | Project structure, `TRACKER.md`, `PROJECT_OVERVIEW.md` | Empty placeholders define the planned modules and directories; implementation has not started. |
| 2026-10-24 | Added Python `.gitignore` rules for caches, environments, build output, notebooks, and OS/editor files. | `.gitignore`, `TRACKER.md` | Generated local files stay out of version control while source and benchmark artifacts remain available to commit. |


