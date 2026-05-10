-- LLM Inference Benchmarking Suite — Postgres schema
--
-- Loaded automatically on first container startup via
-- /docker-entrypoint-initdb.d. Idempotent — safe to re-run.
--
-- Schema mirrors CSV schema v2: every row records identity, configuration,
-- workload, metrics, and library versions. This makes the table directly
-- queryable in Grafana without joins.

CREATE TABLE IF NOT EXISTS benchmark_results (
    id              SERIAL PRIMARY KEY,

    -- Identity
    runner_name     TEXT        NOT NULL,           -- 'vllm' | 'hf'
    model_alias     TEXT        NOT NULL,           -- 'tinyllama' | 'mistral-7b' | etc.
    model_id        TEXT        NOT NULL,           -- HuggingFace ID

    -- Configuration the row was generated under
    kernel          TEXT        NOT NULL,           -- 'awq' | 'awq_marlin' | 'none' | 'hf_none'
    quantization    TEXT        NOT NULL,
    dtype           TEXT,
    max_model_len   INTEGER,

    -- Workload
    prompt_set      TEXT        NOT NULL,           -- 'short' | 'medium' | 'long'
    batch_size      INTEGER     NOT NULL,
    max_new_tokens  INTEGER,

    -- Latency (milliseconds)
    ttft_ms             DOUBLE PRECISION,
    mean_itl_ms         DOUBLE PRECISION,
    p90_itl_ms          DOUBLE PRECISION,
    p99_itl_ms          DOUBLE PRECISION,
    total_latency_ms    DOUBLE PRECISION,

    -- Throughput
    output_tokens       INTEGER,
    throughput_tok_s    DOUBLE PRECISION,
    input_tokens        INTEGER,

    -- Memory (megabytes)
    gpu_mem_used_mb     DOUBLE PRECISION,
    gpu_mem_reserved_mb DOUBLE PRECISION,

    -- Reproducibility
    vllm_version            TEXT,
    transformers_version    TEXT,
    torch_version           TEXT,
    schema_version          TEXT,

    -- When the benchmark row was generated (from the CSV row, not insertion time)
    measured_at TIMESTAMP    NOT NULL,

    -- When this database row was inserted (for tracking re-ingestion)
    ingested_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes match the most common Grafana query patterns:
-- "filter by model + runner + kernel" and "group by batch_size + prompt_set"
CREATE INDEX IF NOT EXISTS idx_results_model_runner_kernel
    ON benchmark_results (model_alias, runner_name, kernel);

CREATE INDEX IF NOT EXISTS idx_results_workload
    ON benchmark_results (batch_size, prompt_set);

CREATE INDEX IF NOT EXISTS idx_results_measured_at
    ON benchmark_results (measured_at DESC);

-- Convenience view: the latest run for each (runner, kernel, model, prompt_set, batch).
-- Re-running a benchmark inserts new rows; this view shows only the most recent.
CREATE OR REPLACE VIEW v_latest_results AS
SELECT DISTINCT ON (runner_name, kernel, model_alias, prompt_set, batch_size)
    *
FROM benchmark_results
ORDER BY runner_name, kernel, model_alias, prompt_set, batch_size, measured_at DESC;