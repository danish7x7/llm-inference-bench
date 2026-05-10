"""
hf_runner.py — HuggingFace Transformers baseline backend.

Uses the standard AutoModelForCausalLM + generate() API.
No continuous batching, no paged attention — this is the "naive" baseline
that vLLM should significantly outperform on throughput.

Spawn-safe: torch/transformers imports are deferred to load().
"""

from __future__ import annotations

import time
from typing import List, Optional

from .base import InferenceRunner, GenerationConfig, RunResult
from ..metrics import MetricsCollector, GPUMemorySnapshot


class HFRunner(InferenceRunner):
    """
    HuggingFace Transformers baseline.

    Characteristics being measured (vs vLLM):
      - No continuous batching: each generate() call blocks until done
      - Standard KV cache (not paged): memory fragmentation expected
      - Static batching: all sequences in a batch run to max_new_tokens
      - No CUDA graph optimization by default
    """

    def __init__(
        self,
        model_id: str,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "auto",              # "auto" | "float16" | "bfloat16"
        load_in_4bit: bool = False,       # bitsandbytes 4-bit quantization
        load_in_8bit: bool = False,
        attn_implementation: str = "eager",  # "eager" | "flash_attention_2" | "sdpa"
        apply_chat_template: bool = True,
        min_new_tokens: int = 32,
    ):
        super().__init__(model_id, gpu_memory_utilization)
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.attn_implementation = attn_implementation
        self.apply_chat_template = apply_chat_template
        self.min_new_tokens = min_new_tokens
        self._model = None
        self._tokenizer = None
        self._device = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

        print(f"[hf] Loading {self.model_id} ...")
        t0 = time.perf_counter()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, "auto")

        # ------------------------------------------------------------------
        # Quantization handling:
        #   1. If the model is *already* quantized (AWQ/GPTQ baked into
        #      config.json), HF detects it and loads via the matching path.
        #      We must NOT pass our own bnb config or HF rejects the call.
        #   2. If the user requested bnb 4/8-bit AND the model is NOT
        #      pre-quantized, build a BitsAndBytesConfig.
        # ------------------------------------------------------------------
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=False)
        is_pre_quantized = hasattr(config, "quantization_config") and config.quantization_config

        quant_config = None
        if is_pre_quantized:
            existing = config.quantization_config
            quant_method = existing.get("quant_method") if isinstance(existing, dict) else getattr(existing, "quant_method", "unknown")
            print(f"[hf] Model is pre-quantized ({quant_method}); using HF native path, ignoring load_in_4bit/8bit flags")
        elif self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=False,
        )
        # Ensure pad token exists (needed for batched generation)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            device_map="auto",
            attn_implementation=self.attn_implementation,
            trust_remote_code=False,
        )
        self._model.eval()

        elapsed = time.perf_counter() - t0
        self._loaded = True
        print(f"[hf] Model loaded in {elapsed:.1f}s")

    def unload(self) -> None:
        import gc
        import torch
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        print(f"[hf] {self.model_id} unloaded")

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def run(
        self,
        prompts: List[str],
        config: GenerationConfig,
        prompt_set_name: str = "unknown",
    ) -> RunResult:
        if not self._loaded or self._model is None:
            raise RuntimeError("Call load() before run()")

        assert len(prompts) == config.batch_size

        # Apply chat template if model has one
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

        # Tokenize — left-pad for batched generation
        self._tokenizer.padding_side = "left"
        inputs = self._tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

        input_token_count = inputs["input_ids"].shape[0] * inputs["input_ids"].shape[1]

        # Common generation kwargs
        gen_kwargs = dict(
            max_new_tokens=config.max_new_tokens,
            min_new_tokens=min(self.min_new_tokens, config.max_new_tokens),
            do_sample=(config.temperature > 0),
            temperature=config.temperature if config.temperature > 0 else None,
            top_p=config.top_p if config.temperature > 0 else None,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        # Dispatch: streaming path for batch=1, two-phase path for batch>1
        # (HF's TextIteratorStreamer fundamentally only supports batch=1)
        if config.batch_size == 1:
            collector, output_token_count, ttft_quality = self._run_streaming(
                inputs, gen_kwargs
            )
        else:
            collector, output_token_count, ttft_quality = self._run_batched(
                inputs, gen_kwargs, config.batch_size
            )

        mem = GPUMemorySnapshot.capture()

        # ------------------------------------------------------------------
        # Compute metrics from collector — shared between both paths
        # ------------------------------------------------------------------
        total_wall_ms = (collector.t_end - collector.t_start) * 1000.0
        ttft_ms = (collector.t_first_token - collector.t_start) * 1000.0

        itl_list_ms: List[float] = []
        prev = collector.t_first_token
        for t in collector.token_times:
            itl_list_ms.append((t - prev) * 1000.0)
            prev = t
        itl_list_ms = [x for x in itl_list_ms if x > 0]

        mean_itl = sum(itl_list_ms) / len(itl_list_ms) if itl_list_ms else 0.0
        p90_itl = _percentile(itl_list_ms, 90) if itl_list_ms else 0.0
        p99_itl = _percentile(itl_list_ms, 99) if itl_list_ms else 0.0

        throughput = output_token_count / (total_wall_ms / 1000.0) if total_wall_ms > 0 else 0.0

        result = RunResult(
            runner_name="hf",
            model_id=self.model_id,
            prompt_set=prompt_set_name,
            batch_size=config.batch_size,
            ttft_ms=round(ttft_ms, 2),
            mean_itl_ms=round(mean_itl, 2),
            p90_itl_ms=round(p90_itl, 2),
            p99_itl_ms=round(p99_itl, 2),
            total_latency_ms=round(total_wall_ms, 2),
            output_tokens=output_token_count,
            throughput_tok_s=round(throughput, 2),
            gpu_mem_used_mb=round(mem.used_mb, 1),
            gpu_mem_reserved_mb=round(mem.reserved_mb, 1),
            input_tokens=input_token_count,
        )
        # Document how TTFT/ITL were obtained for this row.
        # "streamed" = direct per-token timing; "two_phase" = TTFT from prefill probe,
        # ITL averaged from total decode time.
        result.notes = f"ttft_quality={ttft_quality}"
        return result

    # ------------------------------------------------------------------
    # Path 1: batch=1 — use TextIteratorStreamer for true per-token timing
    # ------------------------------------------------------------------

    def _run_streaming(self, inputs, gen_kwargs):
        """Per-token streaming path. Only valid for batch_size == 1."""
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        kwargs = {**gen_kwargs, "streamer": streamer}

        collector = MetricsCollector()
        first_token_time: List[float] = []

        collector.start()
        thread = Thread(
            target=self._model.generate,
            kwargs={"input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    **kwargs},
            daemon=True,
        )
        thread.start()

        token_count = 0
        for text_chunk in streamer:
            if token_count == 0:
                first_token_time.append(time.perf_counter())
            token_count += 1
            collector.record_token()

        thread.join()
        collector.stop()

        if first_token_time:
            collector.t_first_token = first_token_time[0]
        else:
            collector.t_first_token = collector.t_start

        return collector, token_count, "streamed"

    # ------------------------------------------------------------------
    # Path 2: batch>1 — two-phase generate for real TTFT, then full run
    # ------------------------------------------------------------------

    def _run_batched(self, inputs, gen_kwargs, batch_size):
        """
        Two-phase batched path:
          Phase A: generate exactly 1 new token to measure pure prefill cost (= TTFT).
          Phase B: generate the full max_new_tokens to measure total throughput.

        ITL is then derived as (total_time - prefill_time) / (output_tokens - 1),
        which gives an *average* decode latency. We mark ttft_quality="two_phase"
        so this is traceable in the output CSV.

        Why a separate prefill probe instead of just (total - some_constant)?
          - Prefill cost depends on input length × batch size. A constant fraction
            heuristic (e.g. "TTFT is 12% of total") collapses across these axes
            and washes out the prefill-vs-decode story we want to tell.
          - One extra forward pass per batch is cheap (~ms on TinyLlama, ~50ms on 7B).
        """
        import torch

        # ---- Phase A: prefill probe (1 new token) ----
        # min_new_tokens must be removed/relaxed so we can stop after exactly 1 token
        prefill_kwargs = {**gen_kwargs}
        prefill_kwargs["max_new_tokens"] = 1
        prefill_kwargs.pop("min_new_tokens", None)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_start = time.perf_counter()

        with torch.no_grad():
            _ = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **prefill_kwargs,
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_first_token = time.perf_counter()

        # ---- Phase B: full generation (timed end-to-end) ----
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_end = time.perf_counter()

        # Count actually generated tokens (excludes input)
        new_token_count = (outputs.shape[1] - inputs["input_ids"].shape[1]) * batch_size

        # Build a synthetic collector: TTFT from phase A, then evenly-distributed
        # decode timestamps from phase B's elapsed time.
        collector = MetricsCollector()
        collector.t_start = t_start
        collector.t_first_token = t_start + (t_first_token - t_start)  # phase A duration
        collector.t_end = collector.t_first_token + (t_end - t_first_token)

        # Distribute remaining decode tokens evenly across phase B's wall time
        decode_tokens = max(new_token_count - batch_size, 1)  # subtract the 1 already-counted prefill token per sequence
        decode_duration = t_end - t_first_token
        if decode_tokens > 0 and decode_duration > 0:
            step = decode_duration / decode_tokens
            collector.token_times = [
                collector.t_first_token + step * (i + 1)
                for i in range(decode_tokens)
            ]
        else:
            collector.token_times = []

        return collector, new_token_count, "two_phase"


def _percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)