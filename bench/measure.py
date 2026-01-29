"""
    Purpose: accurate GPU timing (P50/P95), warmup, synchronization, and a clean JSON writer
"""

import json
import os
import time
from typing import Callable, Dict, Any, List

import numpy as np
import torch


def _ensure_dir_for_files(path: str) -> None:
    """Create parent directory for a file path if it does not exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    _ensure_dir_for_files(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def summarize_stats(values: List[float]) -> Dict[str, float]:
    """Compute p50, p95, mean, std for a list of floats. Used by benchmarks and LLM bench."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def cuda_time_ms(fn: Callable[[], Any], iters: int = 100, warmup: int = 20) -> Dict[str, float]:
    """
    Measure GPU execution time for fn() in milliseconds using CUDA events.
    GPU kernels are async; CUDA events time on the GPU timeline.
    Returns summary stats: p50_ms, p95_ms, mean_ms, std_ms (all in ms).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not detected")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = np.empty(iters, dtype=np.float64)

    for i in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times[i] = starter.elapsed_time(ender)

    return {
        "iters": float(iters),
        "warmup": float(warmup),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std(ddof=1)) if iters > 1 else 0.0,
    }


def cpu_time_ms(fn: Callable[[], Any], iters: int = 50, warmup: int = 5) -> Dict[str, float]:
    """
    CPU wall-clock timing (ms). Use for CPU work only.
    For GPU kernels, prefer cuda_time_ms().
    """
    for _ in range(warmup):
        fn()

    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times[i] = (t1 - t0) * 1000.0

    return {
        "iters": float(iters),
        "warmup": float(warmup),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std(ddof=1)) if iters > 1 else 0.0,
    }


def device_info() -> Dict[str, Any]:
    """Return minimal device + software info to attach to artifacts."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
        })
    return info
