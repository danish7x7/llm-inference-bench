# LLM Inference Benchmarking Suite — Changelog

Daily log of work done. **Format:** Each entry has Date / What I Did / What Worked / What Broke / Tomorrow's First Task.

This file is for two purposes:
1. Continuity — when I pick this back up after a break, the latest entry tells me exactly where I left off
2. AI context — when I start a new chat with Claude, I paste the latest 1–3 entries so it has full context
3. Instructions to be in the env and then start making project
---

## 2026-04-27 — Day 1 (Monday)

**What I did:**
- Verified WSL2 Ubuntu setup, NVIDIA RTX 4060 detected via `nvidia-smi` (CUDA 12.7, driver 566.24, 8GB VRAM)
- Major C: drive cleanup: deleted Anaconda3 (6.4GB), Python 3.13 standalone, Python 3.12 Microsoft Store install, disabled all Python execution aliases, cleared pip cache, emptied Recycle Bin
- C: free space: 22.8GB → ~50GB
- Uninstalled Wuthering Waves from D: drive — D: free space 97GB → 248GB
- Decided to STAY on C: drive WSL (sufficient space, skip migration)
- Installed Miniconda inside WSL Ubuntu (`conda 26.1.1`, base Python 3.13.12)
- Created project directory `~/projects/llm-inference-bench`
- Created `.gitignore`, copied PROJECT_PLAN.md / CHANGELOG.md / README.md from D:\Projects\Inference-bench
- Initialized git, configured user.name=danish7x7, user.email=danishbirsinghbhatti@gmail.com
- First commit: `09e31f5 Initial commit` (4 files)
- Accepted conda Terms of Service for main + r channels
- Created `vllm` conda environment with Python 3.13
- Verified Python 3.13 install in vllm env at `/home/danish07/miniconda3/envs/vllm/bin/python`

**What worked:**
- All cleanup steps clean — Windows still healthy
- WSL Miniconda install perfect
- Git init + commit successful
- vllm env created cleanly

**What broke / Notes:**
- Initial confusion about Python 3.13 + vLLM — verified vLLM 0.19+ supports 3.13
- Conda required ToS acceptance for default channels (one-time)

## 2026-04-27 — Day 1 wrap (Monday afternoon)

**What I did:**
- Installed vLLM 0.20.0 — required cu129-tagged wheel from GitHub releases (PyPI default is now cu130, mismatched my driver)
- Installed `build-essential` (gcc) — needed by Triton/Inductor for torch.compile JIT
- First successful inference: OPT-125M generated coherent text, 43 tok/s output
- Saved hello world to `scripts/hello_vllm.py`

**What worked:**
- `uv pip install <github-wheel-url> --extra-index-url https://download.pytorch.org/whl/cu129` got the right cu129 vLLM build
- `if __name__ == "__main__":` guard required for vLLM under WSL (forces spawn, not fork)
- gpu_memory_utilization=0.5 leaves plenty of headroom on 8GB; got 96K KV cache tokens

**What broke / Notes:**
- vLLM PyPI default switched to cu130 in 0.20.0; bare `pip install vllm` ships a wheel that needs libcudart.so.13. Fix: install +cu129 wheel from GH releases.
- WSL forces multiprocessing start method to spawn; entry-point scripts MUST be guarded
- Triton needs gcc on the system (not bundled). `apt install build-essential` fixes it.
- Persistent benign warning: `destroy_process_group() was not called before program exit` — known vLLM cleanup quirk, not a real issue

**Numbers from first run (OPT-125M, 8GB RTX 4060 Laptop, max_model_len=512):**
- Model load: 1.28s (cached)
- torch.compile: 4.12s (cached)
- Engine init total: 16.7s
- Output throughput: ~43 tok/s (warm, batch=2)

**Environment state at end of Day 1:**
- vllm 0.20.0+cu129, torch 2.11.0+cu129, Python 3.13.13, gcc 13.x
- Conda env `vllm` at /home/danish07/miniconda3/envs/vllm
- Working dir: ~/projects/llm-inference-bench
- HF cache: ~/.cache/huggingface (OPT-125M downloaded, ~250MB)
- vLLM compile cache: ~/.cache/vllm/torch_compile_cache (warm)

---

## 2026-05-09 — Day 2 + Day 3 combined session (Saturday, big jump after a gap)

Came back to the project after ~2 weeks. One long session that covered everything from
"first benchmark architecture" through "actual real-model results." Much of the value here is
in the sequence of bugs found — keep this for the writeup.

**What I built (Day 2):**
- `src/runners/base.py` — abstract `InferenceRunner` interface, `GenerationConfig` and `RunResult` dataclasses, spawn-safe (no CUDA imports at module load)
- `src/runners/vllm_runner.py` — vLLM backend using offline `LLM` class, extracts per-request timing from `RequestMetrics`, falls back to wall-time heuristics
- `src/runners/hf_runner.py` — HuggingFace baseline. **Two-path dispatch** in `run()`:
  - `batch_size=1` → uses `TextIteratorStreamer` for true per-token timing
  - `batch_size>1` → two-phase generate (1-token probe for TTFT, then full generation; ITL averaged across decode time). HF streamers fundamentally cannot batch.
- `src/metrics.py` — `MetricsCollector`, `GPUMemorySnapshot` (pynvml + torch fallback), `compute_metrics()`. Computes TTFT, mean/p90/p99 ITL, throughput, VRAM.
- `benchmark.py` — CLI entry point. `MODEL_REGISTRY` maps short aliases (`tinyllama`, `mistral-7b`, `llama-3-8b`) to HF IDs and per-model settings. Sweeps prompt sets × batch sizes, writes CSV.
- `scripts/smoke_test.py` — quick sanity check on TinyLlama before any real run
- `prompts/{short,medium,long}_prompts.json` — 8/6/3 prompts of escalating length

**Bugs hit and fixed (in order — this is the sequence for the writeup):**

1. **Module import path bug.** `Path(__file__).parent.parent` in `smoke_test.py` resolved correctly only when invoked from project root. Fix: `Path(__file__).resolve().parent.parent`.

2. **Same path bug in `benchmark.py`** but inverted — `benchmark.py` is at the project root, so `.parent.parent` walked one level too high. Fix: `PROJECT_ROOT = Path(__file__).resolve().parent` (no second `.parent`).

3. **Chat template missing — model emitted 1 token then stopped.** First smoke test produced 1 output token, ~1 tok/s "throughput." Cause: TinyLlama-Chat is chat-tuned; we sent raw text, it saw an unfinished sentence and emitted EOS. Fix in both runners:
    - Load `AutoTokenizer` alongside the model
    - Apply `tokenizer.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True)` before generation
    - Add `min_tokens=32` (vLLM `SamplingParams`) and `min_new_tokens=32` (HF `generate`) so benchmark always produces enough output to measure
    - Confirmation signal: vLLM load message now says `(chat_template=yes)`

4. **HF runner crashed at batch>1: `TextStreamer only supports batch size 1`.** HF's streamer fundamentally cannot batch. Fix: dispatch by batch size (see `_run_streaming` and `_run_batched` in hf_runner.py). Two-phase batched path measures real TTFT via a `max_new_tokens=1` probe, then runs full generation. ITL becomes an average (p90/p99 collapse to the mean) but TTFT is honest. Trade-off documented in `result.notes = "ttft_quality=two_phase"`.

5. **Mistral-7B AWQ crashed: KV cache shortfall on 8GB.** vLLM allocations: weights 3.88GB + CUDA graphs 1.46GB + activations leaves only 0.4GB for KV cache, while max_model_len=4096 needs 0.5GB. Fix: dropped `max_model_len` from 4096 → 2048 in registry for all 7-8B models. Our actual prompts max out around ~600 tokens so this is fine.

6. **🚨 BIG ONE: vLLM was using the wrong AWQ kernel.** First Mistral-7B run showed throughput of ~6.7 tok/s — clearly broken (TinyLlama at 1/6 the size was 100+ tok/s). Buried in load logs: `[awq_marlin.py] Detected that the model can run with awq_marlin, however you specified quantization=awq explicitly, so forcing awq. Use quantization=awq_marlin for faster inference`. The generic AWQ kernel is dramatically slower than the Marlin kernel which uses tensor cores properly. **Fix: changed `"quantization": "awq"` → `"quantization": "awq_marlin"` in registry. Result: 9× speedup across the board.** Saved both CSVs (`vllm_mistral-7b_awq_generic.csv` and `vllm_mistral-7b_*.csv` post-fix) for the writeup comparison.

**What worked / data we now have:**

vLLM TinyLlama-1.1B fp16 (batch 1→4, medium prompt): **111 → 217 → 368 tok/s**
HF TinyLlama-1.1B fp16 (batch 1→4, medium prompt): **53 → 101 → 196 tok/s**
→ vLLM ~2× HF on throughput, validating the resume bullet.

vLLM Mistral-7B AWQ-Marlin (batch 1→4, medium prompt): **59 → 116 → 233 tok/s**
vLLM Mistral-7B AWQ-generic (batch 1→4, medium prompt): **6.7 → 12.4 → 24.9 tok/s**
→ Marlin gives ~9× speedup over generic AWQ on Ada.

TTFT scales with input length but is **invariant to batch size**:
- short (8 input tok): 300ms across batch 1, 2, 4
- medium (24 input tok): 660ms across batch 1, 2, 4
→ Prefill is parallel, decode amortizes — textbook prefill-vs-decode story for Day 5 writeup.

ITL drops as batch grows: 14ms → 7ms → 3.6ms (Mistral medium).
→ Memory-bound decode amortized across the batch. The "continuous batching is 2x+ throughput improvement" claim is now **real, measured, and reproducible**.

**Known limitations / debt to address later:**

- VRAM reporting is wrong for vLLM. `pynvml` reads GPU-wide usage, but vLLM pre-allocates its KV pool at startup → CSV shows constant ~7100 MB across all batch sizes for Mistral. Fix on Day 4: read vLLM's own KV-cache utilization metrics or compute analytically.
- HF batch>1 ITL p90/p99 collapse to mean (by design — two-phase path averages decode). Documented in `result.notes`.
- TTFT measurement methodology differs between backends (vLLM uses `RequestMetrics`, HF uses streamer first-token or two-phase probe). Honest in writeup, but methodology needs to be explicit.
- Default cudagraph capture sizes go up to 512 (we only sweep 1–4). Reclaiming that 1.46GB is possible but would require passing `compilation_config` to `LLM()`. Defer.

**Where we are right now (end of session):**

✅ Day 1: Environment + vLLM install
✅ Day 2: Runner architecture, both backends working with chat templates
✅ Day 3 (mostly): Real model (Mistral-7B AWQ-Marlin), full prompt × batch sweep on vLLM done

⏳ **Immediately next: HF baseline for Mistral-7B.** Same matrix, gives the cleanest vLLM-vs-HF comparison number for the headline ("vLLM ~9× HF on 7B AWQ"). HF + bitsandbytes 4bit will be slow — expect 30-45 min runtime, throughput in 5-15 tok/s range.

⏳ Day 4 then: PostgreSQL + Grafana + Docker Compose, results dashboard
⏳ Day 5: README, writeup, polish, push to GitHub

**Files in current state:**
- `benchmark.py` — at project root, with PROJECT_ROOT path fix and Marlin in registry
- `src/runners/{base,vllm_runner,hf_runner}.py` — chat templates + min_tokens applied, HF dispatch on batch size
- `src/metrics.py`, `src/config.py`, `src/__init__.py`
- `prompts/{short,medium,long}_prompts.json`
- `scripts/smoke_test.py`
- `requirements.txt`
- `results/` contains:
  - `vllm_tinyllama_*.csv`
  - `hf_tinyllama_*.csv` (×2 — one before HF batched fix, one after)
  - `vllm_mistral-7b_awq_generic.csv` (renamed from original) — KEEP, for writeup comparison
  - `vllm_mistral-7b_*.csv` (post-Marlin) — KEEP, this is our headline data

**Next session — first task:**
```bash
cd ~/projects/llm-inference-bench
conda activate vllm
python benchmark.py --model mistral-7b --runner hf --prompts short medium long --batch-sizes 1 2 4
```
Expected: ~30-45 min. Will write `results/hf_mistral-7b_*.csv`. Watch for OOM on batch=4 medium/long — if so, try `--batch-sizes 1 2` first or drop to short only.

After HF baseline, decide between (a) Llama-3 8B (skip if Mistral story is sufficient — likely is) or (b) jump straight to Day 4 infra.

**Environment state:**
- vllm 0.20.0+cu129, torch 2.11.0+cu129, Python 3.13.13, gcc 13.x
- Conda env `vllm`
- HF cache populated: TinyLlama, Mistral-7B AWQ (~4GB), OPT-125M
- vLLM compile caches warm for: TinyLlama, Mistral-7B (both kernels)
- Working dir: ~/projects/llm-inference-bench
- Marlin kernel: confirmed working, 9× over generic AWQ