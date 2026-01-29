"""
    Purpose: single source of truth for resume metrics — gather all artifacts into summary.json and summary.csv
"""

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
    """Load JSON file if it exists; return None otherwise."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.getLogger(__name__).warning("Could not load %s: %s", path, e)
        return None


def build_summary(
    llm_path: Path,
    layernorm_path: Path,
) -> Dict[str, Any]:
    """
    Build summary dict from artifact paths.
    LLM: tokens/sec P50, latency P50, peak VRAM P50, model name, GPU name.
    LayerNorm (if present): baseline P50/P95 ms, optimized P50/P95 ms, speedup P50/P95, max error.
    """
    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "llm": None,
        "layernorm": None,
    }

    llm = _load_json(llm_path)
    if llm and llm.get("kind") == "llm_infer_bench":
        device = llm.get("device") or {}
        summary["llm"] = {
            "tokens_per_sec_p50": llm.get("tokens_per_sec", {}).get("p50"),
            "latency_p50_s": llm.get("latency_seconds", {}).get("p50"),
            "peak_vram_mb_p50": llm.get("peak_mem_mb", {}).get("p50"),
            "model": llm.get("model"),
            "gpu_name": device.get("gpu_name"),
        }

    ln = _load_json(layernorm_path)
    if ln and ln.get("kind") == "layernorm_bench":
        summary["layernorm"] = {
            "baseline_p50_ms": ln.get("baseline_ms", {}).get("p50"),
            "baseline_p95_ms": ln.get("baseline_ms", {}).get("p95"),
            "optimized_p50_ms": ln.get("optimized_ms", {}).get("p50"),
            "optimized_p95_ms": ln.get("optimized_ms", {}).get("p95"),
            "speedup_p50": ln.get("speedup", {}).get("p50"),
            "speedup_p95": ln.get("speedup", {}).get("p95"),
            "max_abs_error": ln.get("max_abs_error"),
        }

    return summary


def summary_to_csv_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten summary into one row per section for CSV (one row with all headline fields)."""
    row: Dict[str, Any] = {
        "timestamp_utc": summary.get("timestamp_utc", ""),
        "gpu_name": "",
        "llm_model": "",
        "llm_tokens_per_sec_p50": "",
        "llm_latency_p50_s": "",
        "llm_peak_vram_mb_p50": "",
        "layernorm_baseline_p50_ms": "",
        "layernorm_baseline_p95_ms": "",
        "layernorm_optimized_p50_ms": "",
        "layernorm_optimized_p95_ms": "",
        "layernorm_speedup_p50": "",
        "layernorm_speedup_p95": "",
        "layernorm_max_abs_error": "",
    }

    llm = summary.get("llm") or {}
    row["gpu_name"] = llm.get("gpu_name") or ""
    row["llm_model"] = llm.get("model") or ""
    row["llm_tokens_per_sec_p50"] = llm.get("tokens_per_sec_p50") if llm.get("tokens_per_sec_p50") is not None else ""
    row["llm_latency_p50_s"] = llm.get("latency_p50_s") if llm.get("latency_p50_s") is not None else ""
    row["llm_peak_vram_mb_p50"] = llm.get("peak_vram_mb_p50") if llm.get("peak_vram_mb_p50") is not None else ""

    ln = summary.get("layernorm") or {}
    row["layernorm_baseline_p50_ms"] = ln.get("baseline_p50_ms") if ln.get("baseline_p50_ms") is not None else ""
    row["layernorm_baseline_p95_ms"] = ln.get("baseline_p95_ms") if ln.get("baseline_p95_ms") is not None else ""
    row["layernorm_optimized_p50_ms"] = ln.get("optimized_p50_ms") if ln.get("optimized_p50_ms") is not None else ""
    row["layernorm_optimized_p95_ms"] = ln.get("optimized_p95_ms") if ln.get("optimized_p95_ms") is not None else ""
    row["layernorm_speedup_p50"] = ln.get("speedup_p50") if ln.get("speedup_p50") is not None else ""
    row["layernorm_speedup_p95"] = ln.get("speedup_p95") if ln.get("speedup_p95") is not None else ""
    row["layernorm_max_abs_error"] = ln.get("max_abs_error") if ln.get("max_abs_error") is not None else ""

    return [row]


def write_csv(path: Path, summary: Dict[str, Any]) -> None:
    """Write summary as a single-row CSV (headline metrics)."""
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
        description="Gather benchmark artifacts into summary.json and summary.csv"
    )
    ap.add_argument(
        "--llm",
        type=Path,
        default=Path("artifacts/llm.json"),
        help="Path to llm_infer_bench artifact",
    )
    ap.add_argument(
        "--layernorm",
        type=Path,
        default=Path("artifacts/layernorm.json"),
        help="Path to layernorm_bench artifact (optional)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for summary.json and summary.csv",
    )
    args = ap.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("Building summary from %s, %s", args.llm, args.layernorm)

    summary = build_summary(args.llm, args.layernorm)
    json_path = args.out_dir / "summary.json"
    csv_path = args.out_dir / "summary.csv"

    write_json(str(json_path), summary)
    write_csv(csv_path, summary)

    logger.info("Wrote %s and %s", json_path, csv_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
