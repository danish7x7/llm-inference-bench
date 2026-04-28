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

**Next task (Day 1 — Monday Apr 27):**
1. `cd ~/projects/llm-inference-bench`
2. `conda activate vllm`
3. Run `pip install vllm` (10-15 min, ~6-8 GB download)
4. Verify install: `python -c "import vllm; print(vllm.__version__)"`
5. Hello world test with OPT-125M model
6. If hello world works → start writing `src/runners/base.py` and `src/runners/vllm_runner.py`
7. Commit progress

**Open decisions:**
- All resolved as of end of Day 1

**Environment state:**
- WSL Ubuntu, conda env `vllm` created with Python 3.13
- Project at `~/projects/llm-inference-bench` (also accessible via `\\wsl$\Ubuntu\home\danish07\projects\llm-inference-bench`)
- Git initialized, 1 commit made
- vLLM NOT YET INSTALLED — that's the next task today

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

**Next task (Day 2 — Tue Apr 28):**
1. Write `src/runners/base.py` — abstract `InferenceRunner` interface (must be spawn-safe)
2. Write `src/runners/vllm_runner.py` implementing the interface
3. Write `src/metrics.py` — TTFT, ITL, throughput, GPU memory (pynvml)
4. Test on TinyLlama 1.1B before moving to bigger models on Day 3

**Environment state:**
- vllm 0.20.0+cu129, torch 2.11.0+cu129, Python 3.13.13, gcc 13.x
- Conda env `vllm` at /home/danish07/miniconda3/envs/vllm
- Working dir: ~/projects/llm-inference-bench
- HF cache: ~/.cache/huggingface (OPT-125M downloaded, ~250MB)
- vLLM compile cache: ~/.cache/vllm/torch_compile_cache (warm)

