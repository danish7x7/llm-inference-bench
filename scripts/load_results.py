"""
load_results.py — Ingest all CSVs from results/ into Postgres.

Usage:
    python scripts/load_results.py              # append all rows
    python scripts/load_results.py --truncate   # wipe table first, then insert

Postgres connection settings:
    Reads PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD env vars.
    Defaults match docker-compose.yml: localhost:5432, bench/bench, db=benchmarks.

Run order:
    1. docker compose up -d         (Postgres + Grafana running)
    2. python scripts/load_results.py
    3. Open http://localhost:3000   (dashboard auto-loads)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage import ingest_directory, truncate_table, DEFAULT_DSN

RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--truncate", action="store_true",
        help="Wipe benchmark_results table before inserting (clean slate)"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help=f"Directory containing CSVs (default: {RESULTS_DIR})"
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: results dir not found: {args.results_dir}")
        sys.exit(1)

    csvs = list(args.results_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {args.results_dir}")
        sys.exit(0)

    print(f"Connecting to Postgres at {DEFAULT_DSN['host']}:{DEFAULT_DSN['port']}/{DEFAULT_DSN['dbname']}")
    print(f"Found {len(csvs)} CSV file(s) in {args.results_dir}\n")

    if args.truncate:
        print("Truncating benchmark_results table...")
        truncate_table()
        print("  ✓ Table cleared\n")

    print("Ingesting CSVs:")
    stats = ingest_directory(args.results_dir)

    print(f"\nDone. {stats.files_processed} files processed, "
          f"{stats.rows_inserted} rows inserted, "
          f"{stats.rows_skipped} rows skipped.")
    if stats.errors:
        print(f"Errors: {stats.errors}")
        sys.exit(2)

    print("\nNext: open http://localhost:3000")


if __name__ == "__main__":
    main()