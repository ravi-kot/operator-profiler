"""
    Purpose: end-to-end LLM inference benchmark (latency, tokens/sec, peak VRAM) with reproducible JSON artifacts
"""

import argparse
import logging
import time
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from bench.utils import (
    setup_logging,
    run_main,
    require_cuda,
    validate_positive_int,
    validate_nonempty,
)
from bench.measure import write_json, device_info, summarize_stats


def bench_llm(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    repeats: int,
    warmup_runs: int,
    dtype: str,
) -> Dict[str, Any]:

    logger = logging.getLogger(__name__)

    logger.info(f"Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)

    # Map dtype string to torch dtype (keep simple, predictable)
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError("dtype must be one of: fp16, bf16, fp32")

    logger.info(f"Loading model on CUDA: {model_name} | dtype={dtype}")
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
        .to("cuda")
        .eval()
    )

    logger.info("Tokenizing prompt and moving to CUDA")
    try:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = prompt
    x = tok(prompt_text, return_tensors="pt").to("cuda")

    # Warmup: stabilize first-run effects (allocators, caches, kernel selection)
    logger.info(f"Warmup runs: {warmup_runs}")
    for _ in range(warmup_runs):
        _ = model.generate(**x, max_new_tokens=min(32, max_new_tokens), do_sample=False)
    torch.cuda.synchronize()

    lat_s: List[float] = []
    tps: List[float] = []
    peak_mb: List[float] = []
    gen_tokens_list: List[int] = []

    logger.info(f"Benchmark repeats: {repeats}")
    for r in range(repeats):

        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        y = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        prompt_tokens = int(x["input_ids"].shape[1])
        total_tokens = int(y.shape[1])
        gen_tokens = total_tokens - prompt_tokens

        dt = float(t1 - t0)
        tokens_per_sec = float(gen_tokens / dt) if dt > 0 else 0.0
        peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

        logger.info(
            f"[run {r+1}/{repeats}] gen_tokens={gen_tokens} "
            f"latency={dt:.3f}s tokens/s={tokens_per_sec:.2f} peak_mem={peak_mem_mb:.0f}MB"
        )

        lat_s.append(dt)
        tps.append(tokens_per_sec)
        peak_mb.append(peak_mem_mb)
        gen_tokens_list.append(gen_tokens)

    out = {
        "kind": "llm_infer_bench",
        "model": model_name,
        "dtype": dtype,
        "prompt_chars": len(prompt),
        "prompt_tokens": int(x["input_ids"].shape[1]),
        "max_new_tokens": max_new_tokens,
        "repeats": repeats,
        "warmup_runs": warmup_runs,
        "gen_tokens_each_run": gen_tokens_list,
        "tokens_per_sec": summarize_stats(tps),
        "latency_seconds": summarize_stats(lat_s),
        "peak_mem_mb": summarize_stats(peak_mb),
        "device": device_info(),
    }

    return out


def main() -> None:

    ap = argparse.ArgumentParser(description="LLM inference benchmark (tokens/sec, latency, peak VRAM).")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompt", default="Explain GPU operator performance in 5 bullet points.")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup_runs", type=int, default=2)
    ap.add_argument("--dtype", default="fp16", help="fp16 | bf16 | fp32")
    ap.add_argument("--out", default="artifacts/llm.json")

    args = ap.parse_args()

    require_cuda()
    validate_nonempty("model", args.model)
    validate_nonempty("prompt", args.prompt)
    validate_positive_int("max_new_tokens", args.max_new_tokens)
    validate_positive_int("repeats", args.repeats)
    validate_positive_int("warmup_runs", args.warmup_runs)
    validate_nonempty("dtype", args.dtype)

    logging.getLogger(__name__).info("Starting LLM inference benchmark...")
    out = bench_llm(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        dtype=args.dtype,
    )

    write_json(args.out, out)
    logging.getLogger(__name__).info(f"Wrote artifact: {args.out}")

    print(out)


if __name__ == "__main__":
    setup_logging()
    run_main(main)
 