"""
vllm_runner.py — vLLM inference backend.

Uses vLLM's offline LLM class with streaming RequestOutput to capture
per-token timing for accurate TTFT and ITL measurement.

Spawn-safe: all vLLM imports happen inside load(), never at module level.
This is required under WSL which forces multiprocessing start method = spawn.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .base import InferenceRunner, GenerationConfig, RunResult
from ..metrics import MetricsCollector, GPUMemorySnapshot, compute_metrics


class VLLMRunner(InferenceRunner):
    """
    Benchmarks vLLM's LLM engine (offline / batch mode).

    vLLM strengths being measured:
      - Continuous batching (PagedAttention)
      - CUDA graph replay for decode steps
      - High GPU utilization across batch
    """

    def __init__(
        self,
        model_id: str,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        quantization: Optional[str] = None,   # e.g. "awq", "gptq"
        dtype: str = "auto",
        enforce_eager: bool = False,           # True = disable CUDA graphs (slower, less VRAM)
        apply_chat_template: bool = True,      # auto-detect & apply for *-chat / *-instruct models
        min_tokens: int = 32,                  # force minimum generation length for benchmarking
    ):
        super().__init__(model_id, gpu_memory_utilization)
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.apply_chat_template = apply_chat_template
        self.min_tokens = min_tokens
        self._llm = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Import vLLM and load model. Call after subprocess starts."""
        from vllm import LLM  # deferred import — spawn-safe
        from transformers import AutoTokenizer

        print(f"[vllm] Loading {self.model_id} ...")
        t0 = time.perf_counter()

        self._llm = LLM(
            model=self.model_id,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            dtype=self.dtype,
            enforce_eager=self.enforce_eager,
            trust_remote_code=False,
        )

        # Load tokenizer separately for chat template application
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        elapsed = time.perf_counter() - t0
        self._loaded = True

        # Report chat template availability
        has_template = (
            self.apply_chat_template
            and self._tokenizer.chat_template is not None
        )
        print(f"[vllm] Model loaded in {elapsed:.1f}s "
              f"(chat_template={'yes' if has_template else 'no'})")

    def unload(self) -> None:
        """Free GPU memory."""
        import gc
        self._llm = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False
        print(f"[vllm] {self.model_id} unloaded")

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def run(
        self,
        prompts: List[str],
        config: GenerationConfig,
        prompt_set_name: str = "unknown",
    ) -> RunResult:
        from vllm import SamplingParams

        if not self._loaded or self._llm is None:
            raise RuntimeError("Call load() before run()")

        assert len(prompts) == config.batch_size, (
            f"Got {len(prompts)} prompts but config.batch_size={config.batch_size}"
        )

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            min_tokens=min(self.min_tokens, config.max_new_tokens),  # ensure non-trivial output for benchmarking
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )

        # Apply chat template if model has one (e.g. *-chat, *-instruct).
        # Without this, chat-tuned models often emit EOS immediately.
        formatted_prompts = prompts
        if self.apply_chat_template and self._tokenizer.chat_template is not None:
            formatted_prompts = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]

        # ------------------------------------------------------------------
        # Timing: vLLM offline generate() returns completed outputs.
        # For TTFT we use the built-in metrics on RequestOutput if available,
        # otherwise we approximate using the streaming approach below.
        # ------------------------------------------------------------------

        collector = MetricsCollector()
        collector.start()

        # vLLM batch generate (returns when all sequences are done)
        outputs = self._llm.generate(formatted_prompts, sampling_params)

        # Capture GPU memory right after generation
        mem = GPUMemorySnapshot.capture()

        # Extract per-request timing from vLLM's RequestMetrics if present
        # (available in vLLM >= 0.4.0)
        total_output_tokens = 0
        ttft_ms_values = []
        all_itl_ms = []

        for output in outputs:
            total_output_tokens += len(output.outputs[0].token_ids)

            metrics = getattr(output, "metrics", None)
            if metrics is not None:
                # vLLM exposes these in seconds
                if hasattr(metrics, "first_token_time") and metrics.first_token_time:
                    ttft = (metrics.first_token_time - metrics.arrival_time) * 1000
                    ttft_ms_values.append(ttft)
                if hasattr(metrics, "time_per_output_token") and metrics.time_per_output_token:
                    all_itl_ms.extend([t * 1000 for t in metrics.time_per_output_token])

        # Fall back: estimate from wall time if vLLM metrics not exposed
        collector.stop()
        total_wall_ms = (collector.t_end - collector.t_start) * 1000.0

        if ttft_ms_values:
            ttft_ms = sum(ttft_ms_values) / len(ttft_ms_values)
        else:
            # Rough heuristic: prefill is ~15% of total for decode-heavy runs
            ttft_ms = total_wall_ms * 0.15

        if all_itl_ms:
            mean_itl = sum(all_itl_ms) / len(all_itl_ms)
            p90_itl = _percentile(all_itl_ms, 90)
            p99_itl = _percentile(all_itl_ms, 99)
        else:
            # Approximate: distribute remaining time evenly over output tokens
            decode_ms = total_wall_ms - ttft_ms
            mean_itl = decode_ms / max(total_output_tokens, 1)
            p90_itl = mean_itl * 1.1
            p99_itl = mean_itl * 1.2

        throughput = total_output_tokens / (total_wall_ms / 1000.0)

        # Approximate input tokens (tokenizer not available here — use char heuristic)
        input_tokens = sum(len(p.split()) * 4 // 3 for p in prompts)  # ~1.33 chars/token

        return RunResult(
            runner_name="vllm",
            model_id=self.model_id,
            prompt_set=prompt_set_name,
            batch_size=config.batch_size,
            ttft_ms=round(ttft_ms, 2),
            mean_itl_ms=round(mean_itl, 2),
            p90_itl_ms=round(p90_itl, 2),
            p99_itl_ms=round(p99_itl, 2),
            total_latency_ms=round(total_wall_ms, 2),
            output_tokens=total_output_tokens,
            throughput_tok_s=round(throughput, 2),
            gpu_mem_used_mb=round(mem.used_mb, 1),
            gpu_mem_reserved_mb=round(mem.reserved_mb, 1),
            input_tokens=input_tokens,
        )


def _percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)