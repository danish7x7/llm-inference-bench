"""
base.py — Abstract InferenceRunner interface.

Every backend (vLLM, HF, future SGLang) must implement this.
Designed to be spawn-safe: no GPU state held at module import time.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationConfig:
    """Sampling / generation parameters passed to every runner."""
    max_new_tokens: int = 256
    temperature: float = 0.0        # 0 = greedy, deterministic
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    batch_size: int = 1             # number of prompts per batch


@dataclass
class RunResult:
    """
    Output from a single benchmark run (one batch, one prompt-set).
    All timing in milliseconds, throughput in tokens/second.
    """
    runner_name: str
    model_id: str
    prompt_set: str                 # "short" | "medium" | "long"
    batch_size: int

    # --- Latency ---
    ttft_ms: float                  # time-to-first-token (ms)
    mean_itl_ms: float              # mean inter-token latency (ms/tok)
    p90_itl_ms: float
    p99_itl_ms: float
    total_latency_ms: float         # wall time for entire generation

    # --- Throughput ---
    output_tokens: int              # total tokens generated across batch
    throughput_tok_s: float         # output_tokens / (total_latency_ms / 1000)

    # --- Memory ---
    gpu_mem_used_mb: float          # peak GPU MB during generation
    gpu_mem_reserved_mb: float      # total reserved by CUDA allocator

    # --- Input ---
    input_tokens: int               # total input tokens across batch

    # --- Extra ---
    errors: List[str] = field(default_factory=list)
    notes: str = ""


class InferenceRunner(abc.ABC):
    """
    Abstract base class for all inference backends.

    Lifecycle:
        runner = VLLMRunner(model_id, **kwargs)
        runner.load()                        # warm up GPU, load weights
        result = runner.run(prompts, config) # benchmark
        runner.unload()                      # free VRAM

    Implementations must be spawn-safe: no CUDA calls at __init__ time.
    Call load() only after the subprocess has started.
    """

    def __init__(self, model_id: str, gpu_memory_utilization: float = 0.85):
        self.model_id = model_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights onto GPU. Called once before benchmarking."""
        ...

    @abc.abstractmethod
    def unload(self) -> None:
        """Release GPU memory. Called after all runs finish."""
        ...

    # ------------------------------------------------------------------
    # Benchmark entry point
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def run(
        self,
        prompts: List[str],
        config: GenerationConfig,
        prompt_set_name: str = "unknown",
    ) -> RunResult:
        """
        Run inference on `prompts` using `config`.

        Args:
            prompts:        List of raw text prompts. len == config.batch_size.
            config:         Generation parameters.
            prompt_set_name: Label stored in RunResult for grouping results.

        Returns:
            RunResult populated with timing + memory stats.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short name used in CSVs and logs."""
        return self.__class__.__name__.replace("Runner", "").lower()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id}, loaded={self._loaded})"
