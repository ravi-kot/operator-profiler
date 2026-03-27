"""
Purpose: run a single end-to-end paged KV cache serving benchmark and export system metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

from bench.measure import device_info, write_json
from bench.utils import run_main, setup_logging, validate_positive_int
from kv_cache_lab.config import CacheConfig, ModelConfig, ServingConfig, WorkloadConfig
from kv_cache_lab.runtime import run_serving_simulation
from kv_cache_lab.workload import generate_burst_requests


def run_benchmark(
    config_name: str,
    scheduler_policy: str,
    cache_cfg: CacheConfig,
    workload_cfg: WorkloadConfig,
    model_cfg: ModelConfig | None = None,
) -> Dict[str, Any]:
    model_cfg = model_cfg or ModelConfig()
    serving_cfg = ServingConfig(
        name=config_name,
        scheduler_policy=scheduler_policy,
        cache_policy="request_aware" if cache_cfg.request_aware_policy else "uniform",
        model=model_cfg,
        cache=cache_cfg,
        workload=workload_cfg,
    )
    requests = generate_burst_requests(workload_cfg)
    result = run_serving_simulation(serving_cfg, requests)
    payload = {
        "kind": result.artifact_kind,
        "device": device_info(),
        "config": result.config,
        "metrics": result.metrics,
        "per_class": result.per_class,
        "scheduler": result.scheduler,
        "cache": result.cache,
        "trace_summary": result.trace_summary,
        "experiment_notes": result.experiment_notes,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Paged KV cache serving benchmark")
    ap.add_argument("--out", default="artifacts/kv_service.json")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--total_requests", type=int, default=96)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--max_batch_size", type=int, default=16)
    ap.add_argument("--kv_mode", default="fp16")
    ap.add_argument("--scheduler", default="reuse_aware", help="reuse_aware | fifo")
    ap.add_argument("--disable_pinning", action="store_true")
    ap.add_argument("--disable_cost_aware", action="store_true")
    ap.add_argument("--disable_request_aware", action="store_true")
    ap.add_argument("--chunked_prefill_tokens", type=int, default=512)
    args = ap.parse_args()

    validate_positive_int("concurrency", args.concurrency)
    validate_positive_int("total_requests", args.total_requests)
    validate_positive_int("block_size", args.block_size)
    validate_positive_int("max_batch_size", args.max_batch_size)

    workload = WorkloadConfig(
        concurrency=args.concurrency,
        total_requests=args.total_requests,
    )
    cache_cfg = CacheConfig(
        block_size_tokens=args.block_size,
        max_batch_size=args.max_batch_size,
        hot_prefix_pinning=not args.disable_pinning,
        request_aware_policy=not args.disable_request_aware,
        reuse_aware_scheduler=args.scheduler == "reuse_aware",
        cost_aware_mode=not args.disable_cost_aware,
        default_kv_mode=args.kv_mode,
        emergency_kv_mode="fp8" if args.kv_mode == "fp16" else args.kv_mode,
        chunked_prefill_tokens=args.chunked_prefill_tokens,
    )

    payload = run_benchmark(
        config_name="advanced" if args.scheduler == "reuse_aware" else "baseline",
        scheduler_policy=args.scheduler,
        cache_cfg=cache_cfg,
        workload_cfg=workload,
    )
    write_json(args.out, payload)
    logging.getLogger(__name__).info("Wrote artifact: %s", args.out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    setup_logging()
    run_main(main)
