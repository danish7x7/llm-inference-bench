# CLAUDE.md — LLM Inference Benchmarking Suite

Auto-loaded every session. Stable rules and facts only. For the evolving
step-by-step roadmap and progress, see PLAN.md (not auto-loaded — open it
when working the plan).

Two parts: (A) how to behave when editing this repo, (B) facts about it.

---

## PART A — Behavioral guidelines

**Tradeoff:** these bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think before coding
Don't assume. Don't hide confusion. Surface tradeoffs.
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity first
Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility"/"configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical changes
Touch only what you must. Clean up only your own mess.
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/vars/functions that YOUR changes made unused; leave
  pre-existing dead code unless asked.
The test: every changed line should trace directly to the request.

### 4. Goal-driven execution
Define success criteria. Loop until verified.
- "Fix the bug" -> "Write a test that reproduces it, then make it pass."
- For multi-step tasks, state a brief plan with a verify step for each.
- This project is also a learning exercise: when you change something,
  briefly explain the *why* in terms of inference mechanics (effect on KV
  cache, batching, memory, latency) — not just the diff.

### 5. GPU-aware workflow (important here)
You cannot see GPU output. When a task needs a real run:
- Write the code AND give the exact command to run.
- Then stop and ask the user to paste the output. Don't assume it worked.

### 6. Working the plan
When asked to work on the roadmap, read PLAN.md, pick the next unchecked
item (or the one the user names), do it per the rules above, and when it's
verified complete, check it off in PLAN.md and add a one-line note of what
was done. Don't check items off before they're verified.

---

## PART B — This project

### What it is
A benchmarking harness measuring and comparing LLM inference performance
across serving backends (vLLM and HuggingFace) on a single consumer GPU.
Records latency (TTFT, inter-token latency, p90/p99), throughput, and memory,
with a versioned, reproducible CSV schema. Results can also land in Postgres
and be visualized in Grafana (docker-compose.yml).

Purpose is twofold: a working real-GPU inference benchmark, AND a portfolio
artifact targeting GPU/ML-systems inference roles (vLLM/SGLang/TensorRT-LLM,
KV caching, paged attention, batching, quantization). Code clarity and correct
methodology matter as much as raw numbers.

### Hardware / environment constraints
- GPU: laptop RTX 4060, 8 GB VRAM — the binding constraint on everything.
  Large models only run heavily quantized (4-bit AWQ), max_model_len capped
  at 2048. Models that fit: opt-125m, tinyllama (1.1B), 4-bit 7-8B.
- WSL2 (Ubuntu) on Windows. Expected (not bugs): vLLM forces spawn
  multiprocessing; pin_memory=False; NVML not fork-compatible.
- CUDA: driver supports 13.x; torch + vLLM are CUDA-13 pip builds. torch and
  vLLM versions are pinned to each other — do not bump torch independently.
- No system CUDA toolkit / nvcc (only pip runtime libs). Anything that JIT-
  compiles CUDA at runtime (FlashInfer sampler, deep_gemm) fails with
  "Could not find nvcc". Worked around — see below.

### Known issues & workarounds (WSL-specific)
Two env vars must be set before vLLM is imported. Currently in the user's
~/.bashrc; moving them into code is a plan item.
- VLLM_USE_FLASHINFER_SAMPLER=0 — FlashInfer's sampler JIT-compiles a CUDA
  kernel needing nvcc (absent). Native PyTorch sampler is fine for single-GPU.
- VLLM_USE_V2_MODEL_RUNNER=0 — the V2 model runner needs UVA, unavailable
  under WSL GPU passthrough. Forcing V1 avoids the "UVA is not available" crash.
- deep_gemm failed to import / "Not enough SMs for max_autotune" — harmless,
  graceful fallback. Ignore.

### Models known to work / not
- tinyllama (TinyLlama-1.1B-Chat) — reliable smoke-test model, fp16.
- Qwen2.5-3B-Instruct-AWQ — works, 4-bit.
- mistral-7b / llama-3-8b registry entries use awq_marlin with 2023-2024
  configs that MAY not load on current vLLM — if they error, it's a config-
  format mismatch, not harness logic.
- HF runner only expected to work on tinyllama.

### How to run
Smoke test:
    python benchmark.py --model tinyllama --runner vllm --prompts short
Batch sweep:
    python benchmark.py --model tinyllama --runner vllm --prompts short --batch-sizes 1 2 4 8 16
Results -> results/<runner>_<model>_<timestamp>.csv (gitignored).

### Code structure
- benchmark.py — CLI entry, model registry, CSV writer, main loop. MUST run
  as __main__ (vLLM spawn requirement under WSL).
- src/runners/base.py — GenerationConfig, runner base class, RunResult.
- src/runners/vllm_runner.py — vLLM backend. Imports vllm INSIDE load()
  (spawn-safe), never at module level.
- src/runners/hf_runner.py — HuggingFace backend.
- src/metrics.py — MetricsCollector, GPUMemorySnapshot, compute_metrics.
- prompts/ — short/medium/long prompt JSON.
- results/ — output CSVs (gitignored).
- docs/ — experiment writeups.
- docker-compose.yml, postgres/, grafana/ — results storage + dashboards.

### Conventions
- CSV schema is versioned (SCHEMA_VERSION in benchmark.py). If you change
  CSV_FIELDS, bump the version; keep reproducibility columns.
- Every row must carry enough context to compare across vLLM upgrades.
- Prefer measuring over guessing. When a number looks off, profile it.
