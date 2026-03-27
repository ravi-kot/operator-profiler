"""
Purpose: compare KV cache quantization modes for memory savings, latency, and reconstruction fidelity.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import torch

from bench.kv_service_bench import run_benchmark
from bench.measure import write_json
from bench.utils import run_main, setup_logging
from kv_cache_lab.config import CacheConfig, WorkloadConfig
from kv_cache_lab.cuda_bridge import dequantize_tensor, extension_status, quantize_tensor


def _fidelity_metrics(mode: str) -> Dict[str, float]:
    torch.manual_seed(17)
    x = torch.randn(128, 256, dtype=torch.float32)
    if mode == "fp16":
        restored = x.to(torch.float16).float()
    elif mode == "fp8":
        row_view = x.reshape(x.shape[0], -1)
        scale = row_view.abs().amax(dim=1).clamp_min(1e-6) / 96.0
        quantized = torch.round(row_view / scale.unsqueeze(1)).clamp(-96, 96)
        restored = (quantized * scale.unsqueeze(1)).reshape_as(x)
    else:
        packed = quantize_tensor(x, "int8")
        restored = dequantize_tensor(packed["quantized"], packed["scale"], x.shape)

    diff = restored - x
    cosine = torch.nn.functional.cosine_similarity(
        restored.reshape(1, -1), x.reshape(1, -1), dim=1
    )[0].item()
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    return {
        "cosine_similarity": cosine,
        "rmse": rmse,
        "max_abs_error": diff.abs().max().item(),
    }


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for mode in ("fp16", "fp8", "int8"):
        payload = run_benchmark(
            config_name=f"kv_{mode}",
            scheduler_policy="reuse_aware",
            cache_cfg=CacheConfig(
                block_size_tokens=16,
                max_batch_size=16,
                hot_prefix_pinning=True,
                request_aware_policy=True,
                reuse_aware_scheduler=True,
                cost_aware_mode=True,
                default_kv_mode=mode,
                emergency_kv_mode=mode,
                chunked_prefill_tokens=512,
            ),
            workload_cfg=WorkloadConfig(
                concurrency=40,
                total_requests=112,
                burst_window_ms=210,
                seed=31,
            ),
        )
        fidelity = _fidelity_metrics(mode)
        rows.append(
            {
                "kv_mode": mode,
                "peak_kv_memory_mb": payload["metrics"]["peak_kv_memory_mb"],
                "throughput_tokens_per_s": payload["metrics"]["throughput_tokens_per_s"],
                "decode_latency_ms_per_token_p50": payload["metrics"]["decode_latency_ms_per_token"]["p50"],
                "cache_hit_rate": payload["metrics"]["cache_hit_rate"],
                "cosine_similarity": fidelity["cosine_similarity"],
                "rmse": fidelity["rmse"],
                "max_abs_error": fidelity["max_abs_error"],
            }
        )
    return {
        "kind": "kv_quantization_experiment",
        "goal": "Show memory savings and any fidelity or latency tradeoff from KV quantization.",
        "cuda_extension": extension_status(),
        "results": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="KV quantization experiment")
    ap.add_argument("--out", default="artifacts/kv_quantization.json")
    args = ap.parse_args()

    payload = run_experiment()
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    setup_logging()
    run_main(main)
