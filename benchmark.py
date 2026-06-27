"""
benchmark.py — CLI entry point for the LLM Inference Benchmarking Suite.

Usage:
    python benchmark.py --model tinyllama --runner vllm --prompts short
    python benchmark.py --model tinyllama --runner hf --prompts short --batch-sizes 1 2 4
    python benchmark.py --model llama-3-8b --runner vllm --prompts all --batch-sizes 1 2 4 8

This script must be run as __main__ (not imported) because vLLM requires
multiprocessing start method = spawn under WSL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import dataclasses
from pathlib import Path
from typing import List

# ---- Model registry -------------------------------------------------------
# Maps short alias → HuggingFace model ID + recommended settings

MODEL_REGISTRY = {
    "opt-125m": {
        "model_id": "facebook/opt-125m",
        "quantization": None,
        "max_model_len": 512,
        "hf_load_in_4bit": False,
    },
    "tinyllama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "quantization": None,
        "max_model_len": 2048,
        "hf_load_in_4bit": False,
    },
    "mistral-7b": {
        "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "quantization": "awq_marlin",   # Marlin kernel ~4-6x faster than generic awq on Ampere/Ada
        "max_model_len": 2048,    # 4096 doesn't fit on 8GB after CUDA graphs (1.46GB) + weights (3.88GB)
        # NOTE: hf_load_in_4bit kept for registry consistency but ignored —
        # this model is pre-quantized AWQ, HF runner detects this and uses native AWQ path.
        # In practice we don't run HF on this model because TheBloke's 2023 AWQ config
        # is incompatible with current transformers; HF benchmarks live on TinyLlama only.
        "hf_load_in_4bit": False,
    },
    "mistral-7b-awq": {
        # Generic-AWQ side of an A/B against the "mistral-7b" (awq_marlin) entry.
        # Every field is identical except quantization, so the only variable is the
        # kernel — this isolates the Marlin speedup from everything else.
        "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "quantization": "awq",
        "max_model_len": 2048,
        "hf_load_in_4bit": False,
    },
    "llama-3-8b": {
        "model_id": "casperhansen/llama-3-8b-instruct-awq",
        "quantization": "awq_marlin",
        "max_model_len": 2048,    # same VRAM constraint as mistral-7b on 8GB
        "hf_load_in_4bit": True,
    },
}

# ---- Prompt sets ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

PROMPT_SETS = {
    "short": "short_prompts.json",
    "medium": "medium_prompts.json",
    "long": "long_prompts.json",
}


def load_prompts(prompt_set: str) -> List[str]:
    filename = PROMPT_SETS[prompt_set]
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    # Expect either list of strings or list of dicts with "prompt" key
    if isinstance(data[0], str):
        return data
    return [item["prompt"] for item in data]


# ---- CSV output -----------------------------------------------------------

# Schema is versioned and self-describing: each row records not just the metric
# but also the runtime context (which kernel, which quantization, which library
# versions). This is necessary for results to be comparable across vLLM updates
# and for reviewers to understand what changed between two CSV files.
CSV_FIELDS = [
    # Identity
    "runner_name", "model_alias", "model_id",
    # Configuration the row was generated under
    "kernel", "quantization", "dtype", "max_model_len",
    # Workload
    "prompt_set", "batch_size", "max_new_tokens",
    # Latency
    "ttft_ms", "mean_itl_ms", "p90_itl_ms", "p99_itl_ms",
    "total_latency_ms",
    # Throughput / size
    "output_tokens", "throughput_tok_s", "input_tokens",
    # Memory
    "gpu_mem_used_mb", "gpu_mem_reserved_mb",
    # Reproducibility
    "vllm_version", "transformers_version", "torch_version",
    "timestamp", "schema_version",
]

SCHEMA_VERSION = "2"   # bump when CSV_FIELDS changes


def _capture_versions() -> dict:
    """Snapshot installed library versions for the row's reproducibility metadata."""
    versions = {"vllm_version": "", "transformers_version": "", "torch_version": ""}
    try:
        import vllm  # type: ignore
        versions["vllm_version"] = getattr(vllm, "__version__", "")
    except Exception:
        pass
    try:
        import transformers
        versions["transformers_version"] = transformers.__version__
    except Exception:
        pass
    try:
        import torch
        versions["torch_version"] = torch.__version__
    except Exception:
        pass
    return versions


def save_result(result, output_csv: Path, metadata: dict) -> None:
    """
    Write a benchmark row.

    `metadata` is a dict of context fields shared across the whole run
    (kernel, quantization, dtype, etc.) — these don't change row-by-row
    so we build them once and pass them in.
    """
    row = dataclasses.asdict(result)
    row.update(metadata)
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    row["schema_version"] = SCHEMA_VERSION
    row.pop("errors", None)
    row.pop("notes", None)

    write_header = not output_csv.exists()
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})

    print(f"  → Saved to {output_csv}")


# ---- Main benchmark loop --------------------------------------------------

def run_benchmark(
    model_alias: str,
    runner_name: str,
    prompt_sets: List[str],
    batch_sizes: List[int],
    max_new_tokens: int,
    output_csv: Path,
    warmup_runs: int = 1,
    gpu_memory_utilization: float = 0.85,
    enforce_eager: bool = False,
) -> None:

    model_cfg = MODEL_REGISTRY[model_alias]
    model_id = model_cfg["model_id"]

    # ---- Instantiate runner ----
    if runner_name == "vllm":
        from src.runners.vllm_runner import VLLMRunner
        runner = VLLMRunner(
            model_id=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=model_cfg["max_model_len"],
            quantization=model_cfg["quantization"],
            enforce_eager=enforce_eager,
        )
    elif runner_name == "hf":
        from src.runners.hf_runner import HFRunner
        runner = HFRunner(
            model_id=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            load_in_4bit=model_cfg["hf_load_in_4bit"],
        )
    else:
        raise ValueError(f"Unknown runner: {runner_name}. Choose: vllm, hf")

    from src.runners.base import GenerationConfig

    print(f"\n{'='*60}")
    print(f"  Model:   {model_id}")
    print(f"  Runner:  {runner_name}")
    print(f"  Prompts: {prompt_sets}")
    print(f"  Batches: {batch_sizes}")
    print(f"{'='*60}\n")

    runner.load()

    # Build run-level metadata (constant across all rows in this invocation).
    # `kernel` is what's actually running — for vLLM this is the quantization config
    # string vLLM resolved (awq, awq_marlin, gptq, none for fp16). For HF the kernel
    # is determined by transformers' loader; we record what we *requested*.
    quantization = model_cfg.get("quantization") or "none"
    kernel = quantization if runner_name == "vllm" else f"hf_{quantization}"
    dtype = model_cfg.get("dtype", "auto")
    versions = _capture_versions()

    metadata = {
        "model_alias": model_alias,
        "kernel": kernel,
        "quantization": quantization,
        "dtype": dtype,
        "max_model_len": model_cfg.get("max_model_len", ""),
        "max_new_tokens": max_new_tokens,
        **versions,
    }

    try:
        for prompt_set in prompt_sets:
            all_prompts = load_prompts(prompt_set)
            print(f"\n--- Prompt set: {prompt_set} ({len(all_prompts)} prompts) ---")

            for batch_size in batch_sizes:
                # Take first batch_size prompts (repeat if fewer available)
                batch_prompts = (all_prompts * batch_size)[:batch_size]

                config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    batch_size=batch_size,
                )

                # Warmup
                if warmup_runs > 0:
                    print(f"  Warmup (batch={batch_size})...", end=" ", flush=True)
                    runner.run(batch_prompts, config, prompt_set_name=prompt_set)
                    print("done")

                # Actual timed run
                print(f"  Benchmarking batch={batch_size}...", end=" ", flush=True)
                result = runner.run(batch_prompts, config, prompt_set_name=prompt_set)
                print(
                    f"  TTFT={result.ttft_ms:.0f}ms  "
                    f"ITL={result.mean_itl_ms:.1f}ms  "
                    f"Throughput={result.throughput_tok_s:.1f}tok/s  "
                    f"VRAM={result.gpu_mem_used_mb:.0f}MB"
                )

                save_result(result, output_csv, metadata)

    finally:
        runner.unload()

    print(f"\n✓ Benchmark complete. Results: {output_csv}\n")


# ---- CLI ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model alias",
    )
    parser.add_argument(
        "--runner",
        required=True,
        choices=["vllm", "hf"],
        help="Inference backend",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=["short", "medium", "long", "all"],
        default=["short"],
        help="Prompt set(s) to run. Use 'all' for all three.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1],
        metavar="N",
        help="Batch sizes to sweep",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per request",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs before timing (0 to skip)",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=0.85,
        dest="gpu_memory_utilization",
        help="Fraction of GPU VRAM vLLM may use",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="vLLM: disable CUDA graphs (slower, saves VRAM)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Defaults to results/<runner>_<model>_<timestamp>.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Expand "all" prompt sets
    prompt_sets = args.prompts
    if "all" in prompt_sets:
        prompt_sets = ["short", "medium", "long"]

    # Output path
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.output:
        output_csv = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_csv = RESULTS_DIR / f"{args.runner}_{args.model}_{ts}.csv"

    run_benchmark(
        model_alias=args.model,
        runner_name=args.runner,
        prompt_sets=prompt_sets,
        batch_sizes=args.batch_sizes,
        max_new_tokens=args.max_new_tokens,
        output_csv=output_csv,
        warmup_runs=args.warmup,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )