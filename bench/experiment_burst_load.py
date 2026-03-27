"""
Purpose: compare baseline FIFO serving against reuse-aware paged KV serving under bursty concurrency.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from bench.kv_service_bench import run_benchmark
from bench.measure import write_json
from bench.utils import run_main, setup_logging
from kv_cache_lab.config import CacheConfig, ModelConfig, WorkloadConfig


def run_experiment() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    for concurrency in (10, 50, 100):
        total_requests = concurrency * 2
        workload = WorkloadConfig(
            concurrency=concurrency,
            total_requests=total_requests,
            burst_window_ms=180,
            seed=concurrency + 11,
        )
        constrained_model = ModelConfig(kv_budget_gb=11.0)

        baseline = run_benchmark(
            config_name="baseline_fifo",
            scheduler_policy="fifo",
            cache_cfg=CacheConfig(
                block_size_tokens=16,
                max_batch_size=16,
                hot_prefix_pinning=False,
                request_aware_policy=False,
                reuse_aware_scheduler=False,
                cost_aware_mode=False,
                default_kv_mode="fp16",
                emergency_kv_mode="fp16",
                chunked_prefill_tokens=0,
            ),
            workload_cfg=workload,
            model_cfg=constrained_model,
        )
        advanced = run_benchmark(
            config_name="reuse_aware_paged",
            scheduler_policy="reuse_aware",
            cache_cfg=CacheConfig(
                block_size_tokens=16,
                max_batch_size=16,
                hot_prefix_pinning=True,
                request_aware_policy=True,
                reuse_aware_scheduler=True,
                cost_aware_mode=True,
                default_kv_mode="fp16",
                emergency_kv_mode="fp8",
                chunked_prefill_tokens=512,
            ),
            workload_cfg=workload,
            model_cfg=constrained_model,
        )

        results.append(
            {
                "concurrency": concurrency,
                "baseline": baseline["metrics"],
                "advanced": advanced["metrics"],
                "advanced_scheduler": advanced["scheduler"],
                "advanced_cache": advanced["cache"],
            }
        )

    return {
        "kind": "burst_load_experiment",
        "goal": "Show that the reuse-aware scheduler plus paged KV cache survives burst load better than FIFO.",
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Burst-load paged KV cache experiment")
    ap.add_argument("--out", default="artifacts/burst_load.json")
    args = ap.parse_args()

    payload = run_experiment()
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    setup_logging()
    run_main(main)
