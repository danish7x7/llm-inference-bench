# Batch-Size Sweep v2 — 16 Distinct Prompts (no prefix-cache inflation)

**Question:** Did prefix caching inflate the throughput curve in the original
batch sweep? Re-run with distinct prompts to measure the true scaling.

## Why re-run
The original sweep (docs/batch-sweep.md) used a 4-prompt short set. At batch 8
and 16 the harness repeats those prompts, so with vLLM's prefix caching on the
repeated prefixes share KV cache and skip recomputation. That makes high-batch
throughput look better than it would on genuinely distinct traffic. This run
swaps in 16 distinct short prompts so no two requests share a prefix.

## Setup
- Model: TinyLlama-1.1B-Chat-v1.0 (fp16, unquantized)
- Engine: vLLM 0.23.0 (V1 model runner), FlashAttention 2 backend
- Hardware: laptop RTX 4060, 8 GB VRAM, WSL2
- Config: max_model_len 2048, gpu_memory_utilization 0.85, CUDA graphs on,
  prefix caching on (vLLM default)
- Workload: 16 distinct short prompts, batch sizes 1, 2, 4, 8, 16, one warmup
  per batch
- Command: python benchmark.py --model tinyllama --runner vllm --prompts short --batch-sizes 1 2 4 8 16

## Results

| Batch | Throughput (tok/s) | TTFT (ms) | Mean ITL (ms) | vs batch 1 (v2) | vs batch 1 (orig) |
|------:|-------------------:|----------:|--------------:|----------------:|------------------:|
| 1     | 100.3              | 72        | 8.5           | 1.0x            | 1.0x              |
| 2     | 190.0              | 68        | 4.5           | 1.89x           | 1.93x             |
| 4     | 267.1              | 191       | 3.2           | 2.66x           | 2.68x             |
| 8     | 413.9              | 194       | 2.0           | 4.13x           | 5.02x             |
| 16    | 606.3              | 359       | 1.4           | 6.04x           | 9.70x             |

VRAM was pinned at ~7.9 GB across all batch sizes — the 8 GB ceiling, fully
reserved by vLLM at startup regardless of batch.

## What changed
- **Scaling: 6.1x vs 9.7x batch1→16.** The two curves track closely up to batch
  4, then diverge: at batch 8 distinct prompts give 4.13x vs 5.02x, and at batch
  16 it's 6.04x vs 9.70x. The gap is the prefix-cache effect — with distinct
  prompts there are no shared prefixes to skip, so every request pays full
  prefill. Prefix caching accounted for roughly 60% of the apparent gain at
  batch 16; the real throughput-from-batching number is ~6x, not ~10x.
- **TTFT profile: a new jump at batch 16.** Both sweeps show TTFT flat (~70 ms)
  at batch 1–2 then a ~190 ms plateau from batch 4. But the original held ~200 ms
  at batch 16, while this run jumps to **359 ms**. With distinct prompts there's
  no cached prefix to short-circuit, so 16 full prefills compete for the same
  pass and each request waits longer for its first token.
- **ITL: unchanged behavior.** Mean ITL falls 8.5→1.4 ms with batch, same as
  before — an aggregate-rate artifact, not a per-request speedup (see below).

## Interpretation (inference mechanics)

**What prefix caching is.** vLLM hashes the token blocks of each prompt's prefix
and keeps their computed KV cache around. If a new request begins with a prefix
that's already in the cache, vLLM reuses those KV blocks instead of recomputing
attention for them — the request effectively starts mid-prefill. This is a pure
win for workloads with shared structure (system prompts, few-shot exemplars,
repeated queries).

**Why repeated prompts flattered the original.** With only 4 distinct prompts, a
batch of 16 is the same 4 prompts ×4. After the first occurrence of each, the
prefix is cached, so 12 of the 16 requests skip most of their prefill. Less
prefill work per pass → more of the GPU's time goes to decode → higher apparent
throughput and a TTFT that barely moves. That's a real optimization, but it
measures *prefix-cache hit rate*, not *batching scaling*.

**Why distinct prompts expose the true cost.** With 16 unique prefixes there are
zero cache hits, so every request runs full prefill. Throughput still scales
(6.1x) because batching amortizes fixed per-pass overhead — kernel launches and
weight reads from HBM — across more sequences. But the gain is smaller and
honest: it's the decode-amortization benefit alone, with no prefill being
skipped.

**What the TTFT jump at batch 16 means: prefill saturation.** TTFT is how long a
request waits for its first token, which is dominated by prefill. Prefill is
compute-bound (a big matmul over all prompt tokens), unlike memory-bound decode.
At small batches the GPU has spare compute and added prefills overlap cheaply, so
TTFT stays flat. By batch 16 the combined prefill work exceeds what one pass can
hide — the prefills serialize against the compute units and each request's
first-token wait climbs (359 ms). The original sweep never hit this wall because
prefix caching kept the actual prefill volume low.

The corrected takeaway: on distinct traffic, batching this model on a 4060 buys
~6x throughput at the cost of ~5x first-token latency (72 → 359 ms). The earlier
~10x figure was prefix caching doing part of the work.

(Mean ITL falling with batch is an aggregate-rate artifact: the harness reports
inter-token latency as total decode time spread across all output tokens from
all requests. More tokens completing per unit time lowers that per-token average
— not a per-request speedup. The user-facing latency signal is TTFT, which rises.)

## Caveats
- Tiny model, short generations: absolute numbers are specific to a 1.1B fp16
  model on a 4060. The shape of the curve is the transferable result.
- Single sample per config: no variance estimate.
- VRAM-bound: at ~7.9 GB used throughout, batch can't grow much further on this
  hardware before OOM.

## What's next
Compare fp16 vs AWQ quantization on the same model where both fit (latency,
throughput, VRAM).
