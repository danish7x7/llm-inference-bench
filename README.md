# LLM Inference Benchmarking Suite

> Reproducible benchmarking framework comparing LLM inference backends on consumer GPU hardware. Measures throughput, time-to-first-token, inter-token latency, and GPU memory utilization across vLLM and Hugging Face Transformers serving — visualized via Grafana, containerized with Docker Compose.

**Status:** 🚧 In active development. README will be finalized when benchmarks are complete.

---

## Why This Exists

Production LLM serving is bottlenecked by GPU memory and inference engine efficiency. While papers tout 10x+ throughput improvements with continuous batching and paged attention, those numbers are typically reported on H100s. This project asks: **what do these optimizations look like on consumer hardware?**

I built this to:
1. Quantify the real-world performance delta between vLLM (state-of-the-art) and vanilla Hugging Face Transformers (baseline) on an RTX 4060
2. Understand where the bottlenecks shift when GPU memory is constrained (8GB)
3. Demonstrate end-to-end ML infrastructure: from raw inference timing → PostgreSQL → Grafana dashboards

---

## Architecture

(Mermaid diagram placeholder — fill in Day 5)

```
[Prompts] → [Benchmark Runner] → [vLLM | HF Backend] → [Metrics Collector]
                                                              ↓
                                                       [PostgreSQL]
                                                              ↓
                                                       [Grafana Dashboard]
```

---

## Hardware

- **GPU:** NVIDIA RTX 4060 Laptop (8GB VRAM)
- **CPU:** AMD Ryzen 5 7640HS
- **OS:** Windows 11 + WSL2 Ubuntu
- **CUDA:** 12.7 (driver 566.24)

---

## Models Benchmarked

| Model | Quantization | VRAM | License |
|---|---|---|---|
| Llama-3 8B Instruct | AWQ 4-bit | ~5 GB | Meta |
| Mistral 7B Instruct | AWQ 4-bit | ~4 GB | Apache 2.0 |
| TinyLlama 1.1B | FP16 | ~2 GB | Apache 2.0 |

---

## Results

(Filled in after Day 3 benchmarking)

| Backend | Model | Throughput (tok/s) | TTFT (ms) | ITL (ms/tok) | GPU Mem (MB) |
|---|---|---|---|---|---|
| vLLM | Llama-3 8B AWQ | TBD | TBD | TBD | TBD |
| HF | Llama-3 8B AWQ | TBD | TBD | TBD | TBD |
| vLLM | Mistral 7B AWQ | TBD | TBD | TBD | TBD |
| HF | Mistral 7B AWQ | TBD | TBD | TBD | TBD |

---

## How To Run

(Filled in Day 4-5)

```bash
git clone https://github.com/danish7x7/llm-inference-bench
cd llm-inference-bench
docker compose up -d            # Starts PostgreSQL + Grafana
./scripts/download_models.sh    # One-time model download
python src/benchmark.py --runner vllm --model llama-3-8b
# View dashboard at http://localhost:3000
```

---

## Technical Write-Ups

- [Prefill vs. Decode: Where Latency Hides in LLM Serving](docs/prefill_vs_decode.md)

---

## Author

Danishbir Singh Bhatti — [LinkedIn](https://linkedin.com/in/danishbir-singh-bhatti) | [Portfolio](https://danishbir-portfolio.vercel.app)

MS Software Engineering @ San José State University (Dec 2025)
