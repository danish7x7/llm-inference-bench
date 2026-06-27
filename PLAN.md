# PLAN.md — Project Roadmap & Progress

Living checklist. Not auto-loaded — open this when working the roadmap.
Rules for keeping it honest:
- Check an item off [x] only when it's verified done (ran, passed, committed).
- When you check something off, add a one-line note of what was actually done
  and any result/number worth remembering.
- Add new items as they come up. Keep the newest "current focus" near the top.

---

## Current focus
> Next up: move the two WSL env-var workarounds into vllm_runner.py (item 2),
> then re-run the batch sweep with distinct prompts (item 4).

---

## Milestones

### Setup & recovery
- [x] Rebuild conda env (llm-bench) from requirements after deletion.
      — Python 3.13, vLLM 0.23.0, CUDA-13 torch 2.11.0.
- [x] Update NVIDIA driver (was 566.24 / CUDA 12.7 -> now CUDA 13.x).
- [x] Verify torch sees the GPU and vLLM loads a model.
      — Qwen2.5-3B-AWQ generated text on first load.
- [x] Verify the full harness end-to-end on tinyllama; CSV written.
      — batch 1: TTFT 77ms, ~94 tok/s, ~7.6GB VRAM.

### Experiments
- [x] Batch-size sweep on tinyllama (1,2,4,8,16). See docs/batch-sweep.md.
      — ~9.7x throughput batch1->16; TTFT plateaus ~200ms from batch 4.
      — Caveat: only 4 distinct prompts + prefix caching may flatter high batch.
- [ ] Re-run batch sweep with >=16 DISTINCT prompts to isolate the
      prefix-caching effect.
      verify: throughput curve recomputed without repeated prefixes.
- [ ] Quantization comparison: fp16 vs AWQ on the same model where both fit.
      Measure latency, throughput, VRAM.
      verify: a docs/ writeup with the three-way tradeoff table.
- [ ] gpu-memory-utilization sweep: how effective KV cache size and max
      concurrency change with utilization (vLLM logs both at startup).
      verify: table of util -> KV cache tokens -> max concurrency.

### Code / robustness
- [ ] Move WSL env-var workarounds into src/runners/vllm_runner.py as
      os.environ.setdefault(...) at module top, before any vllm import.
      verify: env -u VLLM_USE_FLASHINFER_SAMPLER -u VLLM_USE_V2_MODEL_RUNNER
      python benchmark.py --model tinyllama --runner vllm --prompts short
      still succeeds. Then optionally remove the bashrc lines.
- [ ] Confirm mistral-7b / llama-3-8b registry entries load on current vLLM,
      or update/replace the AWQ configs that don't.
      verify: a clean run on at least one 4-bit 7-8B model.
- [ ] Spin up the Postgres + Grafana stack and confirm results ingest +
      dashboard render.
      verify: a benchmark row visible in Grafana.

### Portfolio
- [x] First experiment writeup (docs/batch-sweep.md).
- [ ] Short README section framing the project for reviewers: what it measures,
      what was learned, the infra debugging (driver/CUDA, UVA-under-WSL,
      nvcc-less JIT) as evidence of real inference-engineering depth.
- [ ] Per-experiment writeups for each completed experiment above.

---

## Parking lot (ideas, not committed)
- Speculative decoding throughput test (if a draft model fits in 8GB).
- Compare vLLM vs HF runner head-to-head on tinyllama.
- Rent a cloud A100 for one run at scale to show the curve on real hardware.
