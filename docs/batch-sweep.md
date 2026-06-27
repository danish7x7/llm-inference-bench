# Batch-Size Sweep — TinyLlama on RTX 4060

**Question:** How does request batching trade throughput against latency on a
single consumer GPU?

## Setup
- Model: TinyLlama-1.1B-Chat-v1.0 (fp16, unquantized)
- Engine: vLLM 0.23.0 (V1 model runner), FlashAttention 2 backend
- Hardware: laptop RTX 4060, 8 GB VRAM, WSL2
- Config: max_model_len 2048, gpu_memory_utilization 0.85, CUDA graphs on,
  prefix caching on (vLLM default)
- Workload: short prompt set, batch sizes 1, 2, 4, 8, 16, one warmup per batch
- Command: python benchmark.py --model tinyllama --runner vllm --prompts short --batch-sizes 1 2 4 8 16

## Results

| Batch | Throughput (tok/s) | TTFT (ms) | Mean ITL (ms) | Throughput vs batch 1 |
|------:|-------------------:|----------:|--------------:|----------------------:|
| 1     | 97.5               | 74        | 8.7           | 1.0x                  |
| 2     | 188.3              | 69        | 4.5           | 1.93x                 |
| 4     | 261.7              | 195       | 3.2           | 2.68x                 |
| 8     | 489.5              | 197       | 1.7           | 5.02x                 |
| 16    | 945.6              | 204       | 0.9           | 9.70x                 |

VRAM was pinned at ~8.0 GB across all batch sizes — the 8 GB ceiling, fully
reserved by vLLM at startup regardless of batch.

## Interpretation

Batching is nearly free throughput, up to a point. Going from batch 1 to 16
yields ~9.7x aggregate throughput. Each forward pass carries fixed overhead
(kernel launches, weight reads from HBM); running more sequences through the
same pass amortizes that overhead across more tokens. Continuous batching
working as intended.

The cost is first-token latency. TTFT is flat (~70 ms) at batch 1–2, then jumps
to a ~200 ms plateau from batch 4 on. As more requests share each prefill pass,
an individual request waits longer for its first token. This is the
prefill-vs-decode tension: batching helps steady-state decode throughput but
lengthens the prefill queue each request sits in.

The core tradeoff: batching bought ~10x throughput at the cost of ~3x
first-token latency. A latency-sensitive deployment (interactive chat) caps
batch low; a throughput-sensitive one (bulk offline generation) pushes it high.
The right batch size is a function of the SLA, not a universal number.

(Mean ITL falling with batch is an aggregate-rate artifact: the harness reports
inter-token latency as total decode time spread across all output tokens from
all requests. More tokens completing per unit time lowers that per-token
average — not a per-request speedup. The user-facing latency signal is TTFT,
which rises.)

## Caveats
- Small, repeated prompts: the short set has only 4 prompts, so batch 8 and 16
  repeat prompts. With prefix caching on, repeated prefixes can share KV cache,
  likely flattering throughput at high batch. A clean re-run needs >=16
  distinct prompts.
- Tiny model, short generations: absolute numbers are specific to a 1.1B fp16
  model on a 4060. The shape of the curve is the transferable result.
- Single sample per config: no variance estimate.
- VRAM-bound: at ~8 GB used throughout, batch can't grow much further on this
  hardware before OOM.

## What's next
Re-run with >=16 distinct prompts to isolate the prefix-caching effect, then
compare fp16 vs AWQ quantization on the same model.
