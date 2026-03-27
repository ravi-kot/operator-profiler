"""
Purpose: single source of truth for the KV cache serving project summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench.measure import write_json


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logging.getLogger(__name__).warning("Could not load %s: %s", path, exc)
        return None


def build_summary(
    service_path: Path,
    burst_path: Path,
    block_sweep_path: Path,
    chunked_path: Path,
    quant_path: Path,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": "KV Cache Serving Lab",
            "tagline": "Paged KV cache, hot-prefix pinning, request-aware policy, reuse-aware scheduling, and cost-aware memory control.",
        },
        "headline": {},
        "experiments": {},
        "key_findings": [],
    }

    service = _load_json(service_path) or {}
    burst = _load_json(burst_path) or {}
    block = _load_json(block_sweep_path) or {}
    chunked = _load_json(chunked_path) or {}
    quant = _load_json(quant_path) or {}

    metrics = service.get("metrics") or {}
    device = service.get("device") or {}
    config = service.get("config") or {}
    cache_cfg = config.get("cache") or {}
    model_cfg = config.get("model") or {}

    summary["headline"] = {
        "gpu_target": model_cfg.get("target_gpu"),
        "runtime_device": device.get("gpu_name") or "CPU fallback in this environment",
        "ttft_ms_p50": metrics.get("ttft_ms", {}).get("p50"),
        "decode_latency_ms_per_token_p50": metrics.get("decode_latency_ms_per_token", {}).get("p50"),
        "throughput_tokens_per_s": metrics.get("throughput_tokens_per_s"),
        "peak_kv_memory_mb": metrics.get("peak_kv_memory_mb"),
        "average_active_kv_pages": metrics.get("average_active_kv_pages"),
        "cache_hit_rate": metrics.get("cache_hit_rate"),
        "prefix_reuse_rate": metrics.get("prefix_reuse_rate"),
        "evictions": metrics.get("evictions"),
        "oom_avoided": metrics.get("oom_avoided"),
        "fragmentation_ratio": metrics.get("fragmentation_ratio"),
        "block_size_tokens": cache_cfg.get("block_size_tokens"),
    }

    summary["experiments"] = {
        "burst_load": burst.get("results", []),
        "block_sweep": block.get("results", []),
        "chunked_prefill": chunked.get("results", []),
        "kv_quantization": quant.get("results", []),
    }

    burst_rows = burst.get("results", [])
    if burst_rows:
        hardest = burst_rows[-1]
        baseline_failed = hardest.get("baseline", {}).get("failed_requests", 0)
        advanced_failed = hardest.get("advanced", {}).get("failed_requests", 0)
        if baseline_failed or advanced_failed:
            summary["key_findings"].append(
                f"At {hardest.get('concurrency')} concurrent requests, reuse-aware serving reduced failed requests from {baseline_failed} to {advanced_failed}."
            )
        else:
            baseline_ttft = hardest.get("baseline", {}).get("ttft_ms", {}).get("p50", 0.0)
            advanced_ttft = hardest.get("advanced", {}).get("ttft_ms", {}).get("p50", 0.0)
            baseline_tps = hardest.get("baseline", {}).get("throughput_tokens_per_s", 0.0)
            advanced_tps = hardest.get("advanced", {}).get("throughput_tokens_per_s", 0.0)
            summary["key_findings"].append(
                f"At {hardest.get('concurrency')} concurrent requests, reuse-aware serving cut TTFT p50 from {baseline_ttft:.0f} ms to {advanced_ttft:.0f} ms and raised throughput from {baseline_tps:.0f} to {advanced_tps:.0f} tok/s."
            )

    block_rows = block.get("results", [])
    if block_rows:
        best = max(block_rows, key=lambda row: row.get("throughput_tokens_per_s", 0.0))
        summary["key_findings"].append(
            f"Block size {best.get('block_size_tokens')} delivered the best throughput in the sweep while exposing the reuse versus fragmentation tradeoff."
        )

    quant_rows = quant.get("results", [])
    if quant_rows:
        fp16 = next((row for row in quant_rows if row.get("kv_mode") == "fp16"), None)
        best_mem = min(quant_rows, key=lambda row: row.get("peak_kv_memory_mb", 1e18))
        if fp16 is not None:
            savings = 100.0 * (
                1.0
                - best_mem.get("peak_kv_memory_mb", 0.0)
                / max(fp16.get("peak_kv_memory_mb", 1.0), 1.0)
            )
            summary["key_findings"].append(
                f"{best_mem.get('kv_mode')} KV mode cut peak KV memory by about {savings:.1f}% versus fp16 in the quantization study."
            )

    return summary


def summary_to_csv_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    row: Dict[str, Any] = {
        "timestamp_utc": summary.get("timestamp_utc", ""),
        "gpu_target": "",
        "runtime_device": "",
        "ttft_ms_p50": "",
        "decode_latency_ms_per_token_p50": "",
        "throughput_tokens_per_s": "",
        "peak_kv_memory_mb": "",
        "average_active_kv_pages": "",
        "cache_hit_rate": "",
        "prefix_reuse_rate": "",
        "evictions": "",
        "oom_avoided": "",
        "fragmentation_ratio": "",
        "top_takeaway": "",
    }

    headline = summary.get("headline") or {}
    row["gpu_target"] = headline.get("gpu_target") or ""
    row["runtime_device"] = headline.get("runtime_device") or ""
    row["ttft_ms_p50"] = headline.get("ttft_ms_p50") or ""
    row["decode_latency_ms_per_token_p50"] = headline.get("decode_latency_ms_per_token_p50") or ""
    row["throughput_tokens_per_s"] = headline.get("throughput_tokens_per_s") or ""
    row["peak_kv_memory_mb"] = headline.get("peak_kv_memory_mb") or ""
    row["average_active_kv_pages"] = headline.get("average_active_kv_pages") or ""
    row["cache_hit_rate"] = headline.get("cache_hit_rate") or ""
    row["prefix_reuse_rate"] = headline.get("prefix_reuse_rate") or ""
    row["evictions"] = headline.get("evictions") or ""
    row["oom_avoided"] = headline.get("oom_avoided") or ""
    row["fragmentation_ratio"] = headline.get("fragmentation_ratio") or ""
    takeaways = summary.get("key_findings") or []
    row["top_takeaway"] = takeaways[0] if takeaways else ""

    return [row]


def write_csv(path: Path, summary: Dict[str, Any]) -> None:
    _ensure_dir(path)
    rows = summary_to_csv_rows(summary)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gather KV cache serving artifacts into summary.json and summary.csv"
    )
    ap.add_argument(
        "--service",
        type=Path,
        default=Path("artifacts/kv_service.json"),
        help="Path to the main kv_service artifact",
    )
    ap.add_argument(
        "--burst",
        type=Path,
        default=Path("artifacts/burst_load.json"),
        help="Path to the burst-load experiment artifact",
    )
    ap.add_argument(
        "--block_sweep",
        type=Path,
        default=Path("artifacts/block_sweep.json"),
        help="Path to the block-size sweep experiment artifact",
    )
    ap.add_argument(
        "--chunked",
        type=Path,
        default=Path("artifacts/chunked_prefill.json"),
        help="Path to the chunked-prefill experiment artifact",
    )
    ap.add_argument(
        "--quant",
        type=Path,
        default=Path("artifacts/kv_quantization.json"),
        help="Path to the quantization experiment artifact",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for summary.json and summary.csv",
    )
    args = ap.parse_args()

    logging.getLogger(__name__).info("Building KV cache summary")
    summary = build_summary(
        service_path=args.service,
        burst_path=args.burst,
        block_sweep_path=args.block_sweep,
        chunked_path=args.chunked,
        quant_path=args.quant,
    )
    json_path = args.out_dir / "summary.json"
    csv_path = args.out_dir / "summary.csv"

    write_json(str(json_path), summary)
    write_csv(csv_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
