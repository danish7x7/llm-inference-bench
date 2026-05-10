"""
backfill_csv_schema.py — upgrade existing v1 CSVs to schema v2.

Schema v2 added: kernel, quantization, dtype, max_model_len, max_new_tokens,
model_alias, vllm_version, transformers_version, torch_version, schema_version.

For existing CSVs (generated under schema v1), we infer these fields from
the filename and the model registry. This is explicit, traceable, and runs
once — output is written next to the original with `.v2` suffix, then you
swap them in.

Usage:
    python scripts/backfill_csv_schema.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import CSV_FIELDS, SCHEMA_VERSION, MODEL_REGISTRY, _capture_versions

RESULTS_DIR = PROJECT_ROOT / "results"


# Map filename → (model_alias, kernel) inferences.
# These are deliberately explicit, not pattern-matched, so the human-readable
# context of each backfilled row is auditable.
KNOWN_FILES = {
    "vllm_tinyllama.csv":                {"model_alias": "tinyllama",   "kernel": "none",        "quantization": "none"},
    "hf_tinyllama.csv":                  {"model_alias": "tinyllama",   "kernel": "hf_none",     "quantization": "none"},
    "vllm_mistral-7b_awq_generic.csv":   {"model_alias": "mistral-7b",  "kernel": "awq",         "quantization": "awq"},
    "vllm_mistral-7b_awq_marlin.csv":    {"model_alias": "mistral-7b",  "kernel": "awq_marlin",  "quantization": "awq_marlin"},
}


def upgrade_row(row: dict, file_metadata: dict, versions: dict) -> dict:
    """Add schema-v2 fields to a v1 row."""
    out = {k: row.get(k, "") for k in CSV_FIELDS}

    # Identity
    out["model_alias"] = file_metadata["model_alias"]

    # Configuration
    out["kernel"] = file_metadata["kernel"]
    out["quantization"] = file_metadata["quantization"]
    out["dtype"] = "auto"

    # Look up max_model_len from registry
    cfg = MODEL_REGISTRY.get(file_metadata["model_alias"], {})
    out["max_model_len"] = cfg.get("max_model_len", "")

    # max_new_tokens — these CSVs were all run with default 256
    out["max_new_tokens"] = 256

    # Versions — use *current* installed versions, but mark this clearly in schema
    # We intentionally use current versions (not "unknown") so the column is never
    # null. The schema_version field signals these came from a backfill, not a fresh run.
    out["vllm_version"] = versions["vllm_version"]
    out["transformers_version"] = versions["transformers_version"]
    out["torch_version"] = versions["torch_version"]

    out["schema_version"] = f"{SCHEMA_VERSION}-backfill"

    return out


def upgrade_file(csv_path: Path, file_metadata: dict, versions: dict) -> Path:
    """Read a v1 CSV, write a v2 CSV next to it with `.v2.csv` suffix."""
    out_path = csv_path.with_suffix(".v2.csv")

    with open(csv_path, "r", newline="") as fin:
        reader = csv.DictReader(fin)
        rows = [upgrade_row(r, file_metadata, versions) for r in reader]

    with open(out_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def main():
    print(f"Backfilling CSVs in {RESULTS_DIR}")
    print(f"Target schema version: {SCHEMA_VERSION} (will tag as '{SCHEMA_VERSION}-backfill')")
    print()

    versions = _capture_versions()
    print("Library versions captured:")
    for k, v in versions.items():
        print(f"  {k}: {v or '(not installed)'}")
    print()

    for filename, file_metadata in KNOWN_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"  SKIP  {filename}: not found")
            continue

        out = upgrade_file(path, file_metadata, versions)
        with open(out) as f:
            new_rows = sum(1 for _ in f) - 1
        print(f"  OK    {filename:<40} → {out.name}  ({new_rows} rows, kernel={file_metadata['kernel']})")

    print()
    print("Next steps:")
    print("  1. Review the .v2.csv files in results/")
    print("  2. If they look right, replace originals:")
    print("     for f in results/*.v2.csv; do mv \"$f\" \"${f/.v2/}\"; done")
    print("  3. Day 4 ingestion will use the new schema.")


if __name__ == "__main__":
    main()