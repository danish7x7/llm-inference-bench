"""
smoke_test.py — Quick sanity check for runners before full benchmark sweep.

Run from project root:
    conda activate vllm
    python scripts/smoke_test.py --runner vllm
    python scripts/smoke_test.py --runner hf

Tests with TinyLlama 1.1B (fast, ~2.2GB VRAM).
Verifies that:
  - Model loads without error
  - At least 1 token is generated
  - TTFT and throughput are non-zero
  - GPU memory is reported
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add project root to path so `src` package is findable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ID   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEST_PROMPT = "What is the capital of France? Answer in one word."


def test_vllm():
    from src.runners.vllm_runner import VLLMRunner
    from src.runners.base import GenerationConfig

    runner = VLLMRunner(
        model_id=MODEL_ID,
        gpu_memory_utilization=0.5,
        max_model_len=512,
        enforce_eager=False,
    )
    runner.load()

    config = GenerationConfig(max_new_tokens=64, temperature=0.0, batch_size=1)
    result = runner.run([TEST_PROMPT], config, prompt_set_name="smoke")
    runner.unload()
    return result


def test_hf():
    from src.runners.hf_runner import HFRunner
    from src.runners.base import GenerationConfig

    runner = HFRunner(
        model_id=MODEL_ID,
        load_in_4bit=False,
    )
    runner.load()

    config = GenerationConfig(max_new_tokens=64, temperature=0.0, batch_size=1)
    result = runner.run([TEST_PROMPT], config, prompt_set_name="smoke")
    runner.unload()
    return result


def print_result(result, runner_name: str):
    print(f"\n{'─'*50}")
    print(f"  Runner:      {runner_name}")
    print(f"  TTFT:        {result.ttft_ms:.0f} ms")
    print(f"  Mean ITL:    {result.mean_itl_ms:.1f} ms/tok")
    print(f"  Throughput:  {result.throughput_tok_s:.1f} tok/s")
    print(f"  Output tok:  {result.output_tokens}")
    print(f"  VRAM used:   {result.gpu_mem_used_mb:.0f} MB")
    print(f"{'─'*50}")

    # Assertions
    assert result.output_tokens > 0,       "No tokens generated!"
    assert result.ttft_ms > 0,             "TTFT is zero!"
    assert result.throughput_tok_s > 0,    "Throughput is zero!"
    print(f"  ✓ All checks passed for {runner_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=["vllm", "hf", "both"], default="vllm")
    args = parser.parse_args()

    runners_to_test = ["vllm", "hf"] if args.runner == "both" else [args.runner]

    for runner_name in runners_to_test:
        print(f"\n[smoke_test] Testing {runner_name} with {MODEL_ID}...")
        if runner_name == "vllm":
            result = test_vllm()
        else:
            result = test_hf()
        print_result(result, runner_name)

    print("✓ Smoke test complete.\n")