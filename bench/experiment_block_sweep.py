"""
Purpose: sweep block sizes to expose the reuse versus fragmentation tradeoff in a paged KV cache.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from bench.kv_service_bench import run_benchmark
from bench.measure import write_json
from bench.utils import run_main, setup_logging
from kv_cache_lab.config import CacheConfig, WorkloadConfig


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for block_size in (8, 16, 32, 64):
        payload = run_benchmark(
            config_name=f"block_{block_size}",
            scheduler_policy="reuse_aware",
            cache_cfg=CacheConfig(
                block_size_tokens=block_size,
                max_batch_size=16,
                hot_prefix_pinning=True,
                request_aware_policy=True,
                reuse_aware_scheduler=True,
                cost_aware_mode=True,
                default_kv_mode="fp16",
                emergency_kv_mode="fp8",
                chunked_prefill_tokens=512,
            ),
            workload_cfg=WorkloadConfig(
                concurrency=48,
                total_requests=128,
                burst_window_ms=220,
                seed=block_size + 19,
            ),
        )
        rows.append(
            {
                "block_size_tokens": block_size,
                "throughput_tokens_per_s": payload["metrics"]["throughput_tokens_per_s"],
                "cache_hit_rate": payload["metrics"]["cache_hit_rate"],
                "prefix_reuse_rate": payload["metrics"]["prefix_reuse_rate"],
                "fragmentation_ratio": payload["metrics"]["fragmentation_ratio"],
                "peak_kv_memory_mb": payload["metrics"]["peak_kv_memory_mb"],
                "decode_latency_ms_per_token_p50": payload["metrics"]["decode_latency_ms_per_token"]["p50"],
            }
        )
    return {
        "kind": "block_size_sweep_experiment",
        "goal": "Find the block size sweet spot between reuse and fragmentation.",
        "results": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Block-size sweep for paged KV cache")
    ap.add_argument("--out", default="artifacts/block_sweep.json")
    args = ap.parse_args()

    payload = run_experiment()
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    setup_logging()
    run_main(main)
