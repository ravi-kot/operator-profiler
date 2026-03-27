"""
Purpose: measure chunked prefill impact on long-context TTFT and burst-time stability.
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
    rows: List[Dict[str, Any]] = []
    constrained_model = ModelConfig(kv_budget_gb=9.0)
    for chunk_size in (0, 256, 512, 1024):
        payload = run_benchmark(
            config_name=f"chunked_prefill_{chunk_size or 'off'}",
            scheduler_policy="reuse_aware",
            cache_cfg=CacheConfig(
                block_size_tokens=16,
                max_batch_size=12,
                hot_prefix_pinning=True,
                request_aware_policy=True,
                reuse_aware_scheduler=True,
                cost_aware_mode=True,
                default_kv_mode="fp16",
                emergency_kv_mode="fp8",
                chunked_prefill_tokens=chunk_size,
            ),
            workload_cfg=WorkloadConfig(
                concurrency=24,
                total_requests=72,
                burst_window_ms=200,
                seed=chunk_size + 23,
                interactive_share=0.30,
                batch_share=0.15,
                long_context_share=0.55,
            ),
            model_cfg=constrained_model,
        )
        rows.append(
            {
                "chunked_prefill_tokens": chunk_size,
                "ttft_ms_p50": payload["metrics"]["ttft_ms"]["p50"],
                "ttft_ms_p95": payload["metrics"]["ttft_ms"]["p95"],
                "throughput_tokens_per_s": payload["metrics"]["throughput_tokens_per_s"],
                "oom_avoided": payload["metrics"]["oom_avoided"],
                "oom_failures": payload["metrics"]["oom_failures"],
                "peak_kv_memory_mb": payload["metrics"]["peak_kv_memory_mb"],
            }
        )
    return {
        "kind": "chunked_prefill_experiment",
        "goal": "Measure long-context stability and TTFT under chunked prefill.",
        "results": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunked-prefill experiment")
    ap.add_argument("--out", default="artifacts/chunked_prefill.json")
    args = ap.parse_args()

    payload = run_experiment()
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    setup_logging()
    run_main(main)
