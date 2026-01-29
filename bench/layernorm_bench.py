"""
    LayerNorm benchmark: PyTorch baseline vs Triton (or F.layer_norm fallback).
    Outputs artifacts/layernorm.json with baseline_ms, optimized_ms, speedup, max_abs_error.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from bench.measure import cuda_time_ms, device_info, write_json
from kernels.triton_layernorm import TRITON_AVAILABLE, triton_layernorm


def bench_layernorm(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    iters: int = 100,
    warmup: int = 20,
    eps: float = 1e-5,
) -> Dict[str, Any]:
    """
    Benchmark PyTorch LayerNorm (baseline) vs triton_layernorm (optimized).
    shape: (batch, seq, hidden) or (batch, hidden); we normalize over the last dim.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for layernorm benchmark")

    hidden = shape[-1]
    norm_shape = (hidden,)
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")
    bias = torch.randn(hidden, dtype=dtype, device="cuda")

    # Correctness: compare baseline vs optimized
    with torch.no_grad():
        y_baseline = F.layer_norm(x, norm_shape, weight, bias, eps)
        y_optimized = triton_layernorm(x, norm_shape, weight, bias, eps)
    max_abs_error = float((y_optimized - y_baseline).abs().max().item())

    # Timing: baseline
    def run_baseline():
        with torch.no_grad():
            F.layer_norm(x, norm_shape, weight, bias, eps)

    baseline_stats = cuda_time_ms(run_baseline, iters=iters, warmup=warmup)

    # Timing: optimized (Triton or F.layer_norm fallback)
    def run_optimized():
        with torch.no_grad():
            triton_layernorm(x, norm_shape, weight, bias, eps)

    optimized_stats = cuda_time_ms(run_optimized, iters=iters, warmup=warmup)

    baseline_p50 = baseline_stats["p50_ms"]
    baseline_p95 = baseline_stats["p95_ms"]
    opt_p50 = optimized_stats["p50_ms"]
    opt_p95 = optimized_stats["p95_ms"]
    speedup_p50 = baseline_p50 / opt_p50 if opt_p50 > 0 else 0.0
    speedup_p95 = baseline_p95 / opt_p95 if opt_p95 > 0 else 0.0

    return {
        "kind": "layernorm_bench",
        "shape": list(shape),
        "dtype": str(dtype),
        "iters": iters,
        "warmup": warmup,
        "triton_available": TRITON_AVAILABLE,
        "baseline_ms": {"p50": baseline_p50, "p95": baseline_p95},
        "optimized_ms": {"p50": opt_p50, "p95": opt_p95},
        "speedup": {"p50": speedup_p50, "p95": speedup_p95},
        "max_abs_error": max_abs_error,
        "device": device_info(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LayerNorm: PyTorch vs Triton benchmark")
    ap.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[4, 512, 1024],
        help="Shape (batch, seq, hidden), e.g. 4 512 1024",
    )
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("artifacts/layernorm.json"))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    shape = tuple(args.shape)
    if len(shape) < 2:
        raise ValueError("shape must have at least (batch, hidden)")
    logger.info("Running LayerNorm benchmark shape=%s (Triton=%s)", shape, TRITON_AVAILABLE)

    out = bench_layernorm(shape=shape, iters=args.iters, warmup=args.warmup)
    write_json(str(args.out), out)
    logger.info("Wrote %s", args.out)
    logger.info(
        "baseline P50=%.3f ms, optimized P50=%.3f ms, speedup P50=%.2fx, max_error=%.2e",
        out["baseline_ms"]["p50"],
        out["optimized_ms"]["p50"],
        out["speedup"]["p50"],
        out["max_abs_error"],
    )
    print(out)


if __name__ == "__main__":
    main()
