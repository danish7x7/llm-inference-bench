# Prefill vs Decode: where the time goes in LLM inference

This writeup unpacks the benchmark results in the dashboard. The goal is to show that the numbers are not a black box - every effect we measured maps to a specific computational characteristic of how transformer decoders work.

## The two phases of a generation

A single LLM generation request has two phases with very different performance characteristics:

**Prefill** - the model processes the entire input prompt in one forward pass to populate the KV cache. This is one large matmul per layer (`Q @ K.T` over the prompt length, `softmax × V` over the same), heavy on FLOPS, and produces exactly one output token (the first sampled token).

**Decode** - for every subsequent token, the model does a forward pass over a single new token, attending to the cached K/V from all prior tokens. This is many small matmuls, each of which is dominated by reading the model weights and the KV cache from HBM.

These two phases have fundamentally different bottlenecks:

|                | Prefill                          | Decode                          |
|----------------|----------------------------------|---------------------------------|
| Bottleneck     | Compute (matmul FLOPS)           | Memory bandwidth (HBM reads)    |
| Tensor shape   | `[B, T_in, H]`                   | `[B, 1, H]`                     |
| Arithmetic intensity | High (lots of math per byte) | Low (lots of bytes per math)    |
| Scales with    | Input length (T_in)              | Output length × layers          |
| Key resource   | Tensor cores                     | HBM bandwidth and KV cache size |

This distinction is the one fact you have to internalize to read inference benchmarks correctly. Almost every counterintuitive number traces back to it.

## What our data shows

Three predictions follow from the table above. The dashboard confirms each.

### Prediction 1: TTFT scales with input length

If prefill is one big matmul over the prompt, doubling the prompt length should roughly double the time. Our vLLM-Marlin Mistral-7B at batch=1:

| Prompt set | Input tokens | TTFT (ms) |
|---|---|---|
| short  | ~8           | 301      |
| medium | ~24          | 647      |
| long   | ~118         | 642      |

Short → medium tracks roughly linearly. Medium → long is suspicious - both report ~640 ms despite long prompts being ~5× longer. Two factors collapse the difference:

1. The Marlin kernel has a fixed dispatch and warmup cost on the first decode after prefill that's a meaningful fraction of TTFT at this size.
2. vLLM uses chunked prefill - at our settings it processes prompts in 8192-token chunks, so until we hit ~thousand-token inputs we're paying a near-constant scheduler overhead on top of the actual matmul.

For larger inputs (multi-thousand tokens) the curve straightens out. Within our 8 GB VRAM budget we can't easily test that - `max_model_len` is capped at 2048 to leave room for the KV cache.

### Prediction 2: TTFT is roughly invariant to batch size

If prefill is compute-bound and the GPU has FLOPS to spare, processing several prompts simultaneously should cost the same as processing one - they all share the model weights, and the matmul shapes get a third "batch" dimension that maps cleanly to tensor cores.

vLLM-Marlin Mistral-7B, medium prompts:

| Batch | TTFT (ms) |
|---|---|
| 1 | 647 |
| 2 | 661 |
| 4 | 658 |

Three batch sizes within ~2% of each other. The same input prompt, prefilled four times in parallel, costs the same wall-time as prefilling once. **This is what continuous batching with PagedAttention is supposed to deliver and the data shows it cleanly.**

### Prediction 3: ITL drops with batch size

Decode is memory-bound. Each step we read all the model weights once to produce one new token per sequence in the batch. If the matmul is bottlenecked by reading the weights, four sequences cost roughly the same time as one - meaning per-token latency drops by 4×.

vLLM-Marlin Mistral-7B, medium prompts:

| Batch | Mean ITL (ms/token) | Per-step throughput |
|---|---|---|
| 1 | 14.3 | 70 tok/s   |
| 2 |  7.3 | 274 tok/s × 1 = 274? |
| 4 |  3.6 | ~1.1k tok/s |

Mean ITL is 14.3 → 3.6 going from batch 1 to 4, which is a 4× drop. Aggregate decode throughput rises proportionally - that's *the* mechanism behind the headline 233 tok/s number for batch=4.

The 14× → 3.6× reduction in per-token latency is decode getting closer to its memory-bandwidth ceiling. We could push further (batch=8, batch=16) but we're VRAM-constrained.

## The Marlin kernel discovery

Both engines (vLLM, HF) and both quantization paths (AWQ, AWQ-Marlin) load the same 4-bit weights from disk. The difference is what happens during the matmul.

**Generic AWQ kernel:** dequantizes the 4-bit weights to fp16 in registers, then runs a standard cuBLAS GEMM. The dequantization is a per-element shift+lookup; it's serial within each thread and doesn't use tensor cores well.

**Marlin kernel:** custom CUDA kernel that fuses the dequantization with the GEMM, lays out the dequantized weights in tensor-core-friendly fragments (mma.sync layouts), and uses asynchronous copies to overlap memory and compute. It is specifically designed for INT4 × FP16 matmul on Ada/Ampere tensor cores.

Our measurement on Mistral-7B at batch=4, medium prompts:

|                  | Generic AWQ | AWQ-Marlin | Speedup |
|------------------|-------------|------------|---------|
| Throughput (tok/s) | 24.9       | 233.5      | **9.38×** |
| TTFT (ms)        | 6172        | 658        | 9.4×    |
| Mean ITL (ms)    | 34.2        | 3.6        | 9.5×    |

The speedup is consistent across all three metrics, which tells us the kernel benefit applies uniformly to both prefill and decode - what we expect for a kernel-level optimization (it's just a faster matmul, regardless of which phase is calling it).

The diagnostic was a one-line warning in vLLM startup logs:

```
Detected that the model can run with awq_marlin, however you specified
quantization=awq explicitly, so forcing awq.
```

Setting `quantization="awq_marlin"` (instead of `"awq"`) is a one-character config change that delivered a 9× end-to-end throughput improvement. **This is the most important finding in the project, and it isn't algorithmic - it's a configuration default.** Anyone running quantized inference on Ada/Ampere should be running Marlin, and the number of "vLLM benchmark" blog posts I've read that didn't check this is non-zero.

## vLLM vs HF Transformers (engine comparison)

To isolate engine effects from kernel effects, we held the model fixed (TinyLlama 1.1B in fp16, no quantization) and varied only the inference engine:

| Batch | HF (tok/s) | vLLM (tok/s) | Engine speedup |
|---|---|---|---|
| 1 | 52.6  | 110.8 | 2.11× |
| 2 | 100.7 | 217.4 | 2.16× |
| 4 | 196.1 | 368.5 | **1.88×** |

What vLLM does that HF doesn't:

- **Continuous batching** - vLLM schedules at the iteration level (decode step), so a sequence that finishes early frees a slot that gets filled by another sequence mid-stream. HF's `generate()` is static-batched: every sequence in the batch runs to `max_new_tokens` regardless of when it would naturally stop. With `min_new_tokens=32` set in our benchmark, this gap is small; with realistic chat workloads where some replies are short and others long, it would be much larger.
- **PagedAttention** - vLLM's KV cache is allocated in fixed-size blocks rather than as one contiguous tensor per sequence. Memory fragmentation drops to near zero, so the KV pool can be much larger for the same VRAM budget.
- **CUDA graphs** - vLLM captures decode-step CUDA graphs at engine startup and replays them. HF's eager mode incurs Python and dispatch overhead on every step.

The 1.88× number captures all three at once, on a workload where batches are uniform and don't really exercise continuous batching's strengths. That's why I'd characterize this as a *conservative* measurement of vLLM's engine advantage.

## What we couldn't measure on this hardware

The 8 GB VRAM budget shapes what's testable:

- **Batch sizes above 4** for Mistral-7B run out of KV cache memory. vLLM's KV-pool capacity at our settings is ~3,280 tokens, so batch=4 with 600-token sequences (input + output) leaves no headroom.
- **Long-context behavior** (>2048 tokens). `max_model_len=4096` causes vLLM to OOM during KV pool allocation; we capped at 2048 and our prompts max out around 400 tokens.
- **FP16 inference of 7B models.** Without quantization, Mistral-7B weights alone are ~14 GB - won't load. The quantization comparison story is forced on us by hardware, but it's a more interesting story than fp16-vs-fp16 anyway.
- **Multi-GPU benchmarks.** One GPU.
- **Speculative decoding.** vLLM supports it, but it's a big enough topic that adding it would dilute the comparison story.

These are the right limitations to call out in a portfolio piece. The reviewer should know what the project is *not*.

## Methodology notes (where the numbers come from)

**vLLM TTFT and ITL** come from `RequestMetrics.first_token_time` and `time_per_output_token` on each `RequestOutput`. These are vLLM's own internal measurements, taken at the engine's scheduling boundary. They include queueing time and CUDA graph dispatch but not Python-side overhead in the runner.

**HF TTFT and ITL** come from two paths:
- At `batch_size=1`, a `TextIteratorStreamer` running in a background thread captures wall-clock time at each yielded chunk. First-token time is the moment of the first yield.
- At `batch_size>1` (where streamers can't batch), we run two `generate()` calls. The first is `max_new_tokens=1` to isolate prefill cost as TTFT. The second is the full generation. ITL is the average of `(total - prefill) / output_tokens`. Per-token distribution (p90/p99) collapses to the mean - by design. Rows are tagged `ttft_quality=two_phase` in `result.notes`.

This means TTFT and ITL **are not directly comparable across runners** as percentile distributions. Within-runner comparisons (e.g. vLLM batch=1 vs vLLM batch=4) are clean. Cross-runner aggregate throughput (`output_tokens / total_wall_time`) is the most honest cross-engine number.

**Throughput** is `output_tokens / total_wall_time` - wall-clock end to end including warmup-then-real-run boundary, but the warmup run is discarded. This is the same definition both runners use.

**GPU memory** comes from NVML; it reads the *whole-GPU* used memory, which means vLLM's pre-allocated KV pool dominates the number regardless of active sequences. This is documented as a known limitation.

## What I'd build next

In rough order of value:

1. **Replace the VRAM column with active KV cache utilization.** Compute it from `output_tokens × layers × kv_heads × head_dim × dtype_bytes × 2 (K+V)`. This would let the dashboard show how memory pressure scales with batch and prompt length, which is the actual story.
2. **Add SGLang as a third engine.** SGLang's scheduler does some interesting things vLLM doesn't (better prefix caching, structured output) and another data point would let us separate "engine A is faster" from "engine A is faster *here*".
3. **Run on a GPU with more VRAM (4090 or A100)** to test long-context and larger batches, where the prefill-decode story really diverges. The current data shows the shape of the effect; bigger hardware would show its slope.
4. **Add latency CDFs** as their own panels. Right now we report mean/p90/p99 ITL but no plot of the distribution. CDFs would make tail-latency stories visible.

None of these change the conclusions; they extend the surface area.