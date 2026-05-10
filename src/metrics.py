"""
metrics.py — Timing and GPU memory measurement utilities.

Design:
  - MetricsCollector wraps a streaming generation loop and records
    the timestamp of every token as it arrives.
  - GPUMemorySnapshot reads NVIDIA NVML for accurate VRAM numbers.
  - compute_metrics() turns raw timestamps into the RunResult fields.

Usage (inside a runner):
    collector = MetricsCollector()
    collector.start()
    for token in stream:
        collector.record_token()
    collector.stop()
    mem = GPUMemorySnapshot.capture()
    result = compute_metrics(collector, mem, ...)
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

# GPU memory measurement.
# Note: pynvml is deprecated in favor of nvidia-ml-py (same API, different PyPI name).
# Run `pip uninstall pynvml -y && pip install nvidia-ml-py` to silence the deprecation warning.
# Both packages expose `import pynvml`, so this code works either way.
try:
    import pynvml  # type: ignore
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# GPU Memory
# ---------------------------------------------------------------------------

@dataclass
class GPUMemorySnapshot:
    used_mb: float       # actually used by tensors
    reserved_mb: float   # total reserved by CUDA allocator (includes fragmentation)
    total_mb: float      # physical VRAM

    @staticmethod
    def capture(device_index: int = 0) -> "GPUMemorySnapshot":
        """Read current VRAM usage via pynvml (preferred) or torch fallback."""
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return GPUMemorySnapshot(
                    used_mb=info.used / 1024**2,
                    reserved_mb=info.used / 1024**2,   # NVML gives used, not reserved
                    total_mb=info.total / 1024**2,
                )
            except Exception:
                pass

        # Fallback: torch (only available if torch is imported in-process)
        try:
            import torch
            if torch.cuda.is_available():
                return GPUMemorySnapshot(
                    used_mb=torch.cuda.memory_allocated(device_index) / 1024**2,
                    reserved_mb=torch.cuda.memory_reserved(device_index) / 1024**2,
                    total_mb=torch.cuda.get_device_properties(device_index).total_memory / 1024**2,
                )
        except Exception:
            pass

        # No GPU measurement available
        return GPUMemorySnapshot(used_mb=0.0, reserved_mb=0.0, total_mb=0.0)


# ---------------------------------------------------------------------------
# Token-level timing collector
# ---------------------------------------------------------------------------

@dataclass
class MetricsCollector:
    """
    Records a timestamp for every token generated.

    Timeline:
        t_start        — just before first forward pass / prefill begins
        t_first_token  — when the first output token arrives (= end of prefill)
        token_times    — wall-clock time of each subsequent token

    From these we derive:
        TTFT  = t_first_token - t_start
        ITL_i = token_times[i] - token_times[i-1]  (or t_first_token for i=0)
    """

    t_start: float = 0.0
    t_first_token: float = 0.0
    t_end: float = 0.0
    token_times: List[float] = field(default_factory=list)
    _started: bool = False

    def start(self) -> None:
        """Call immediately before submitting the request."""
        self.t_start = time.perf_counter()
        self.token_times = []
        self.t_first_token = 0.0
        self._started = True

    def record_token(self) -> None:
        """Call once per output token as it streams out."""
        now = time.perf_counter()
        if not self.token_times and self.t_first_token == 0.0:
            self.t_first_token = now
        self.token_times.append(now)

    def stop(self) -> None:
        """Call after the last token."""
        self.t_end = time.perf_counter()

    # Convenience: record a completed (non-streaming) generation
    def record_completed(self, n_tokens: int, t_end: Optional[float] = None) -> None:
        """
        For non-streaming backends (HF generate()), we don't have per-token times.
        We synthesise evenly-spaced token times between first-token and end.
        TTFT is still measured; ITL is an average approximation.
        """
        if t_end is None:
            t_end = time.perf_counter()
        self.t_end = t_end

        if n_tokens == 0:
            return

        # First token time — best approximation we have without streaming
        if self.t_first_token == 0.0:
            # HF doesn't expose TTFT natively; use a heuristic:
            # assume prefill is ~10% of total time (pessimistic for small models)
            elapsed = t_end - self.t_start
            self.t_first_token = self.t_start + elapsed * 0.1

        # Distribute remaining tokens evenly
        decode_start = self.t_first_token
        decode_end = t_end
        if n_tokens > 1:
            step = (decode_end - decode_start) / n_tokens
            self.token_times = [decode_start + step * i for i in range(n_tokens)]
        else:
            self.token_times = [self.t_first_token]


# ---------------------------------------------------------------------------
# Compute final metrics from collector + memory snapshot
# ---------------------------------------------------------------------------

def compute_metrics(
    collector: MetricsCollector,
    mem: GPUMemorySnapshot,
    runner_name: str,
    model_id: str,
    prompt_set: str,
    batch_size: int,
    input_tokens: int,
) -> dict:
    """
    Turn raw timing data into benchmark metric dict.
    Returns a flat dict (easy to write to CSV / Postgres row).
    """
    total_ms = (collector.t_end - collector.t_start) * 1000.0
    ttft_ms = (collector.t_first_token - collector.t_start) * 1000.0

    # Inter-token latencies
    itl_list_ms: List[float] = []
    prev = collector.t_first_token
    for t in collector.token_times:
        itl_list_ms.append((t - prev) * 1000.0)
        prev = t

    # Filter out zeros / negatives (can happen with batched non-streaming)
    itl_list_ms = [x for x in itl_list_ms if x > 0]

    mean_itl = statistics.mean(itl_list_ms) if itl_list_ms else 0.0
    p90_itl = _percentile(itl_list_ms, 90) if itl_list_ms else 0.0
    p99_itl = _percentile(itl_list_ms, 99) if itl_list_ms else 0.0

    output_tokens = len(collector.token_times)
    throughput = output_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return {
        "runner_name": runner_name,
        "model_id": model_id,
        "prompt_set": prompt_set,
        "batch_size": batch_size,
        "ttft_ms": round(ttft_ms, 2),
        "mean_itl_ms": round(mean_itl, 2),
        "p90_itl_ms": round(p90_itl, 2),
        "p99_itl_ms": round(p99_itl, 2),
        "total_latency_ms": round(total_ms, 2),
        "output_tokens": output_tokens,
        "throughput_tok_s": round(throughput, 2),
        "gpu_mem_used_mb": round(mem.used_mb, 1),
        "gpu_mem_reserved_mb": round(mem.reserved_mb, 1),
        "input_tokens": input_tokens,
    }


def _percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac