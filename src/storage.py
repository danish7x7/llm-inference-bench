"""
storage.py — Postgres ingestion for benchmark CSV files.

Idempotency strategy:
  Each ingestion inserts new rows. The `v_latest_results` view in the
  database picks the most recent row per (runner, kernel, model, prompt, batch),
  so re-running benchmarks doesn't duplicate dashboards — old rows just become
  history. To purge, use TRUNCATE benchmark_results in psql.

Why we don't dedupe on insert:
  The same (runner, kernel, ...) tuple can be measured multiple times — that's
  the *point* of running a benchmark. Storing all runs with `measured_at`
  preserves history. The view handles "show me the current state."
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError as e:
    raise SystemExit(
        "psycopg2 is required for ingestion. Install with:\n"
        "  pip install psycopg2-binary"
    ) from e


# Postgres connection — pulls from env, falls back to docker-compose defaults.
# Override with PGHOST/PGUSER/etc. environment variables for non-default setups.
DEFAULT_DSN = {
    "host":     os.getenv("PGHOST", "localhost"),
    "port":     int(os.getenv("PGPORT", "5432")),
    "dbname":   os.getenv("PGDATABASE", "benchmarks"),
    "user":     os.getenv("PGUSER", "bench"),
    "password": os.getenv("PGPASSWORD", "bench"),
}


# Column order for INSERT — must match (and excludes id, ingested_at which are auto)
DB_COLUMNS = [
    "runner_name", "model_alias", "model_id",
    "kernel", "quantization", "dtype", "max_model_len",
    "prompt_set", "batch_size", "max_new_tokens",
    "ttft_ms", "mean_itl_ms", "p90_itl_ms", "p99_itl_ms", "total_latency_ms",
    "output_tokens", "throughput_tok_s", "input_tokens",
    "gpu_mem_used_mb", "gpu_mem_reserved_mb",
    "vllm_version", "transformers_version", "torch_version", "schema_version",
    "measured_at",
]


@dataclass
class IngestStats:
    files_processed: int = 0
    rows_inserted: int = 0
    rows_skipped: int = 0
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def _coerce(value: str, target_type: type):
    """Convert a CSV string value to the right Python type, handling empty cells."""
    if value is None or value == "":
        return None
    try:
        if target_type is int:
            return int(float(value))     # tolerate "256.0" → 256
        if target_type is float:
            return float(value)
        if target_type is datetime:
            # ISO format with optional timezone — flexible enough for our timestamps
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return target_type(value)
    except (ValueError, TypeError):
        return None


def parse_row(csv_row: dict) -> tuple:
    """
    Convert one CSV row dict into a tuple matching DB_COLUMNS order.
    Returns None if the row is malformed and should be skipped.
    """
    # Required fields — if any are missing, skip the row
    required = ["runner_name", "model_id", "prompt_set", "batch_size", "timestamp"]
    if not all(csv_row.get(k) for k in required):
        return None

    # Build the tuple in DB_COLUMNS order
    return (
        csv_row.get("runner_name"),
        csv_row.get("model_alias") or "unknown",
        csv_row.get("model_id"),
        csv_row.get("kernel") or "unknown",
        csv_row.get("quantization") or "unknown",
        csv_row.get("dtype"),
        _coerce(csv_row.get("max_model_len"), int),
        csv_row.get("prompt_set"),
        _coerce(csv_row.get("batch_size"), int),
        _coerce(csv_row.get("max_new_tokens"), int),
        _coerce(csv_row.get("ttft_ms"), float),
        _coerce(csv_row.get("mean_itl_ms"), float),
        _coerce(csv_row.get("p90_itl_ms"), float),
        _coerce(csv_row.get("p99_itl_ms"), float),
        _coerce(csv_row.get("total_latency_ms"), float),
        _coerce(csv_row.get("output_tokens"), int),
        _coerce(csv_row.get("throughput_tok_s"), float),
        _coerce(csv_row.get("input_tokens"), int),
        _coerce(csv_row.get("gpu_mem_used_mb"), float),
        _coerce(csv_row.get("gpu_mem_reserved_mb"), float),
        csv_row.get("vllm_version"),
        csv_row.get("transformers_version"),
        csv_row.get("torch_version"),
        csv_row.get("schema_version"),
        _coerce(csv_row.get("timestamp"), datetime),
    )


def ingest_csv(path: Path, conn) -> IngestStats:
    """Insert all rows from one CSV file. Returns ingestion stats."""
    stats = IngestStats()
    rows_to_insert = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            parsed = parse_row(row_dict)
            if parsed is None:
                stats.rows_skipped += 1
                continue
            rows_to_insert.append(parsed)

    if rows_to_insert:
        sql = f"""
            INSERT INTO benchmark_results ({', '.join(DB_COLUMNS)})
            VALUES %s
        """
        with conn.cursor() as cur:
            execute_values(cur, sql, rows_to_insert)
        conn.commit()
        stats.rows_inserted = len(rows_to_insert)

    stats.files_processed = 1
    return stats


def ingest_directory(results_dir: Path, conn=None) -> IngestStats:
    """Ingest every *.csv under `results_dir` into Postgres."""
    own_conn = conn is None
    if own_conn:
        conn = psycopg2.connect(**DEFAULT_DSN)

    total = IngestStats()
    try:
        for csv_path in sorted(results_dir.glob("*.csv")):
            stats = ingest_csv(csv_path, conn)
            total.files_processed += stats.files_processed
            total.rows_inserted   += stats.rows_inserted
            total.rows_skipped    += stats.rows_skipped
            total.errors.extend(stats.errors)
            print(f"  ✓ {csv_path.name}: {stats.rows_inserted} rows inserted, {stats.rows_skipped} skipped")
    finally:
        if own_conn:
            conn.close()

    return total


def truncate_table(conn=None) -> None:
    """Wipe all benchmark_results rows. Use before a fresh ingestion."""
    own_conn = conn is None
    if own_conn:
        conn = psycopg2.connect(**DEFAULT_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE benchmark_results RESTART IDENTITY;")
        conn.commit()
    finally:
        if own_conn:
            conn.close()