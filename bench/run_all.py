"""
Purpose: run the full KV cache benchmark suite and refresh dashboard data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from bench.experiment_block_sweep import run_experiment as run_block_sweep
from bench.experiment_burst_load import run_experiment as run_burst_load
from bench.experiment_chunked_prefill import run_experiment as run_chunked_prefill
from bench.experiment_kv_quantization import run_experiment as run_kv_quantization
from bench.kv_service_bench import run_benchmark
from bench.measure import write_json
from bench.summarize import build_summary, write_csv
from kv_cache_lab.config import CacheConfig, WorkloadConfig


def _write_dashboard_summary_js(summary: dict, output_path: Path) -> None:
    output_path.write_text(
        "window.__KV_CACHE_SUMMARY__ = " + json.dumps(summary, indent=2) + ";\n",
        encoding="utf-8",
    )


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    service = run_benchmark(
        config_name="default_suite",
        scheduler_policy="reuse_aware",
        cache_cfg=CacheConfig(),
        workload_cfg=WorkloadConfig(),
    )
    burst = run_burst_load()
    block = run_block_sweep()
    chunked = run_chunked_prefill()
    quant = run_kv_quantization()

    write_json(str(out_dir / "kv_service.json"), service)
    write_json(str(out_dir / "burst_load.json"), burst)
    write_json(str(out_dir / "block_sweep.json"), block)
    write_json(str(out_dir / "chunked_prefill.json"), chunked)
    write_json(str(out_dir / "kv_quantization.json"), quant)

    summary = build_summary(
        service_path=out_dir / "kv_service.json",
        burst_path=out_dir / "burst_load.json",
        block_sweep_path=out_dir / "block_sweep.json",
        chunked_path=out_dir / "chunked_prefill.json",
        quant_path=out_dir / "kv_quantization.json",
    )
    write_json(str(out_dir / "summary.json"), summary)
    write_csv(out_dir / "summary.csv", summary)
    shutil.copyfile(out_dir / "summary.json", Path("dashboard") / "summary.json")
    _write_dashboard_summary_js(summary, Path("dashboard") / "summary.js")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
