# HANDOFF — context for a fresh chat

Paste this (plus CLAUDE.md and PLAN.md) at the start of a new chat to bring a
fresh Claude up to speed. This file is the *story and human context*; the other
two are the project facts and the roadmap.

---

## Who I am and what I'm doing
- I'm an MS-level software engineer with a strong distributed-systems
  background, newer to CUDA / GPU / ML-systems work.
- Goal: build the skills and portfolio to land a GPU / ML-systems inference
  role (target JDs were NVIDIA LLM-inference / training / compiler roles, and a
  YC startup called Luminal). Common thread: CUDA, LLM inference internals (KV
  cache, paged attention, batching, quantization), profiling/optimization, and
  ideally vLLM/SGLang/TensorRT-LLM contributions.
- I want to LEARN by building, not just be handed answers. Willing to spend a
  little money but not much. I work in WSL and use Obsidian for notes.

## The project
- llm-inference-bench — a benchmark harness I built that measures vLLM (and
  HuggingFace) inference performance: TTFT, inter-token latency, throughput,
  memory, with a versioned reproducible CSV schema, plus Postgres + Grafana.
- It's both a real tool and my main portfolio piece. Full details in CLAUDE.md.

## What happened in the last session (so you don't re-tread it)
- I'd accidentally deleted the conda env and caches. We rebuilt everything:
  - Recreated the llm-bench conda env from requirements.txt.
  - NVIDIA driver was way out of date (CUDA 12.7); updated it on the Windows
    side to CUDA 13.x so the CUDA-13 torch/vLLM build works.
  - Fixed two WSL-specific vLLM crashes: FlashInfer sampler needs nvcc (absent)
    -> VLLM_USE_FLASHINFER_SAMPLER=0; V2 model runner needs UVA (broken under
    WSL) -> VLLM_USE_V2_MODEL_RUNNER=0. Both currently live in ~/.bashrc.
  - Verified the harness end-to-end on tinyllama, then ran a batch-size sweep
    (batch 1->16 = ~9.7x throughput, TTFT plateaus ~200ms). Wrote it up in
    docs/batch-sweep.md.
- Set up Claude Code: split context into CLAUDE.md (stable rules/facts, auto-
  loaded) and PLAN.md (living checklist with checkboxes), plus a /experiment
  slash command.

## Where I am right now
- Everything works. Clean, committed state.
- Next planned actions (see PLAN.md for the full list):
  1. Open Claude Code and have it do the first code item: move the two WSL
     env-var workarounds from ~/.bashrc into vllm_runner.py as
     os.environ.setdefault(...) at module top (before any vllm import). Verify:
     env -u VLLM_USE_FLASHINFER_SAMPLER -u VLLM_USE_V2_MODEL_RUNNER
     python benchmark.py --model tinyllama --runner vllm --prompts short
  2. Re-run the batch sweep with >=16 DISTINCT prompts (current sweep used only
     4 prompts, so prefix caching may have flattered high-batch throughput).
  3. fp16 vs AWQ quantization comparison.

## How I like to work with you (chat Claude)
- Help me understand WHAT I'm doing and WHY, not just paste commands — I felt
  lost in the steps at one point, so keep me oriented.
- I'm on WSL; browser downloads land in /mnt/c/Users/danis/Downloads/. I prefer
  creating files directly in the terminal (cat > file << 'EOF') over downloading.
- Give exact commands I can paste. When a step needs GPU output, expect me to
  paste it back rather than you assuming it worked.
- Push back honestly when I'm overcomplicating or cargo-culting.

## Key environment facts (quick reference)
- GPU: laptop RTX 4060, 8 GB VRAM (binding constraint on model size/batch).
- WSL2 + miniconda env llm-bench, Python 3.13, vLLM 0.23.0, CUDA-13 torch.
- Project path: ~/projects/llm-inference-bench
- torch and vLLM versions are pinned together — don't bump torch alone.
