# LLM Inference Benchmarking Suite — Project Plan

**Owner:** Danishbir Singh Bhatti
**Started:** April 26, 2026
**Target completion:** May 1, 2026 (5 working days)
**Purpose:** Resume project demonstrating LLM inference engineering for new-grad SWE/ML roles
**Status:** Day 1 — Environment setup in progress

---

## Goal

Build a reproducible benchmarking framework that compares LLM inference backends (vLLM, Hugging Face Transformers baseline) on quantized open-source models (Llama-3 8B, Mistral 7B), running on consumer GPU hardware (RTX 4060 8GB), measures throughput / TTFT / inter-token latency / GPU memory, stores results in PostgreSQL, and visualizes in Grafana — all containerized with Docker Compose for one-command reproducibility.

## Why This Project

Maps directly to NVIDIA Inference Benchmarking team JD. Demonstrates:
- LLM inference serving knowledge (vLLM internals, paged attention, continuous batching)
- Performance engineering (profiling, benchmarking methodology, bottleneck analysis)
- Production infra (Docker, PostgreSQL, Grafana, reproducibility)
- Technical communication (README, blog post on prefill vs decode trade-offs)

## Scope (In)

- vLLM as primary inference backend
- Hugging Face Transformers as baseline (no continuous batching, no paged attention)
- 2 models: Llama-3 8B Instruct (4-bit quant), Mistral 7B Instruct (4-bit quant)
- Metrics: throughput (tok/s), TTFT (ms), ITL (ms/tok), GPU memory (MB), batch sweep
- Output: CSV → PostgreSQL → Grafana dashboard
- Docker Compose for entire stack
- README + technical write-up on prefill/decode bottlenecks

## Scope (Out — explicitly deferred)

- SGLang (was originally on resume — drop unless time permits Day 5)
- TGI (Hugging Face Text Generation Inference — extra setup overhead)
- Multi-GPU benchmarks (only have 1 GPU)
- Long-context >4K tokens (memory constrained on 8GB VRAM)
- Production serving (we benchmark, we don't deploy)

## Hardware Constraints

- GPU: NVIDIA RTX 4060 Laptop, 8GB VRAM
- Implication: Must use 4-bit quantization (AWQ or GPTQ) to fit 7-8B models
- Implication: Small batch sizes (likely 1-8 max)
- Implication: No FP16 baseline at full size — that's fine, we're showing methodology

## File Structure

```
~/projects/llm-inference-bench/
├── README.md                  # Public-facing
├── PROJECT_PLAN.md            # This file
├── CHANGELOG.md               # Daily log
├── ARCHITECTURE.md            # Tech decisions
├── .gitignore
├── requirements.txt
├── docker-compose.yml         # Day 4
├── Dockerfile                 # Day 4
├── src/
│   ├── benchmark.py
│   ├── runners/{base,vllm_runner,hf_runner}.py
│   ├── metrics.py
│   ├── storage.py
│   └── config.py
├── prompts/{short,medium,long}_prompts.json
├── results/                   # gitignored
├── grafana/dashboards/
├── scripts/
└── docs/prefill_vs_decode.md
```

## Day-by-Day Plan

### Day 1 (Sat Apr 26 — Sun Apr 27): Environment Setup
- [x] Verify NVIDIA GPU + CUDA in WSL Ubuntu (`nvidia-smi`)
- [x] WSL2 confirmed, 953GB virtual disk available
- [x] Free up C: drive (was 22GB → now 49GB free)
- [x] Install Miniconda in WSL
- [ ] Create `vllm` conda env (Python 3.13)
- [ ] `pip install vllm`
- [ ] Hello world test with OPT-125M
- [ ] Git init + first commit

### Day 2 (Mon Apr 28): Core Benchmark Runner
- [ ] Write `runners/base.py` — abstract `InferenceRunner` interface
- [ ] Write `runners/vllm_runner.py` — implements `run(prompts, params)` returning timing data
- [ ] Write `runners/hf_runner.py` — same interface, vanilla transformers
- [ ] Write `metrics.py` — TTFT, ITL, throughput, GPU memory (via pynvml)
- [ ] Write `benchmark.py` — CLI: `python benchmark.py --model llama-3-8b --runner vllm --prompts short`
- [ ] Output: CSV with one row per (runner, model, prompt_set, batch_size) combo
- [ ] Test with TinyLlama 1.1B first (lightweight, fast iteration)

### Day 3 (Tue Apr 29): Real Benchmarks + Quantization
- [ ] Download Llama-3 8B 4-bit AWQ from Hugging Face
- [ ] Download Mistral 7B 4-bit AWQ
- [ ] Run full benchmark sweep across:
  - 2 runners × 2 models × 3 prompt lengths × 4 batch sizes = 48 runs
- [ ] Save results CSV
- [ ] Sanity-check numbers (vLLM should beat HF baseline by 2x+ on throughput)

### Day 4 (Wed Apr 30): Storage + Visualization
- [ ] Write `docker-compose.yml` with PostgreSQL + Grafana services
- [ ] Write `storage.py` — bulk insert benchmark results to Postgres
- [ ] Design Grafana dashboard:
  - Throughput vs batch size (line chart, by runner)
  - TTFT distribution (histogram)
  - GPU memory over time (gauge)
- [ ] Provision dashboard JSON for one-command setup

### Day 5 (Thu May 1): Polish + Ship
- [ ] Write README.md with architecture diagram (Mermaid), results table, how-to-run
- [ ] Write `docs/prefill_vs_decode.md` — technical write-up
- [ ] Containerize benchmark runner in Dockerfile
- [ ] Verify `docker-compose up` brings entire stack up
- [ ] Push to public GitHub: `danish7x7/llm-inference-bench`
- [ ] Add to resume + LinkedIn

### Day 6 (Fri May 2): Visibility
- [ ] LinkedIn post about the project (with screenshots)
- [ ] Optional: Medium/Hashnode blog post adapted from `docs/prefill_vs_decode.md`
- [ ] Apply to 15+ jobs with full 4-project resume

## Key Technical Decisions Made

- **Use vLLM over SGLang/TGI**: Best community support, pip-installable, well-documented
- **Use HF Transformers as baseline**: Universal, no extra setup, fair "naive" comparison
- **4-bit quantization (AWQ)**: Required to fit 7-8B models on 8GB VRAM
- **Python 3.13**: vLLM 0.19+ supports it (verified Apr 26 2026)
- **PostgreSQL over SQLite**: Looks more production, plays well with Grafana
- **Stay on C: drive WSL**: Sufficient space (~50GB free), avoid 30-min migration

## Known Risks

1. **vLLM may not run well on 8GB VRAM** — backup plan: use TinyLlama 1.1B for demonstration, document limitation honestly
2. **Laptop thermal throttling** during long benchmarks — solution: run in chunks, plug in, hard surface
3. **Model downloads timing out** — Llama-3 requires HF account + access request (can take hours/days). Backup: use Qwen 2.5 7B (no auth required)

## Resume Bullet Targets (What This Project Must Justify)

These are the bullets currently on the resume — the project must support each:

1. ✓ "Built end-to-end benchmarking framework evaluating vLLM and HF Transformers across quantized Llama-3 8B and Mistral 7B" — Day 3
2. ✓ "Quantified performance gains from continuous batching and paged attention, demonstrating 2x+ throughput improvements" — Day 3
3. ✓ "Developed live Grafana dashboard backed by PostgreSQL, containerized via Docker Compose for one-command reproducibility" — Day 4
4. ✓ "Authored technical write-up analyzing prefill vs decode bottlenecks and KV-cache memory trade-offs" — Day 5

If any bullet can't be honestly defended by Day 5, **remove it from the resume.**
