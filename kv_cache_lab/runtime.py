from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, List

from kv_cache_lab.cache import PagedKVCache
from kv_cache_lab.config import CacheConfig, ModelConfig, ServingConfig
from kv_cache_lab.scheduler import ReuseAwareScheduler
from kv_cache_lab.types import RequestSpec, RuntimeResult
from kv_cache_lab.workload import build_prefix_catalog


def run_serving_simulation(config: ServingConfig, requests: List[RequestSpec]) -> RuntimeResult:
    cache = PagedKVCache(config.model, config.cache)
    scheduler = ReuseAwareScheduler(config.model, config.cache, config.scheduler_policy)
    cache.pin_hot_prefixes(build_prefix_catalog(), now_ms=0.0)

    current_time_ms = 0.0
    arrivals: Deque[RequestSpec] = deque(sorted(requests, key=lambda req: (req.arrival_ms, req.request_id)))
    ready: List[RequestSpec] = []
    active_page_samples: List[float] = []
    ttft_values: List[float] = []
    decode_latency_values: List[float] = []
    queue_delays: List[float] = []
    throughput_tokens = 0
    completed_requests = 0
    failed_requests = 0
    class_totals: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"ttft": [], "decode": []}
    )

    while arrivals or ready:
        while arrivals and arrivals[0].arrival_ms <= current_time_ms:
            ready.append(arrivals.popleft())

        if not ready:
            current_time_ms = max(current_time_ms, arrivals[0].arrival_ms)
            continue

        ready.sort(key=lambda req: (req.arrival_ms, req.request_id))
        planned_batch = scheduler.plan_batch(ready, cache)
        if not planned_batch:
            current_time_ms += 1.0
            continue

        batch_start_ms = current_time_ms
        if config.scheduler_policy == "reuse_aware":
            batch = _shrink_batch_to_fit(planned_batch, cache, config.model, config.cache)
        else:
            batch = planned_batch

        if not batch:
            current_time_ms += 1.0
            continue

        for req in batch:
            ready.remove(req)

        admitted: List[RequestSpec] = []
        batch_prefill_components: List[float] = []

        for req in batch:
            if not _materialize_request(cache, config, req, batch_start_ms):
                req.failed = True
                req.failure_reason = "kv_budget_exceeded"
                failed_requests += 1
                continue

            admitted.append(req)
            prefix_hit_tokens, prefix_miss_tokens = _prefix_accounting(cache, config.cache, req)
            prefill_ms = _prefill_latency_ms(
                model=config.model,
                cache_cfg=config.cache,
                request=req,
                prefix_hit_tokens=prefix_hit_tokens,
                prefix_miss_tokens=prefix_miss_tokens,
            )
            batch_prefill_components.append(prefill_ms)

        if not admitted:
            current_time_ms += 1.0
            continue

        batch_prefill_ms = max(batch_prefill_components) * (1.0 + 0.05 * max(0, len(admitted) - 1))
        current_time_ms += batch_prefill_ms

        remaining_decode = {req.request_id: req.decode_tokens for req in admitted}
        request_decode_elapsed = {req.request_id: 0.0 for req in admitted}

        for req in admitted:
            queue_delay = batch_start_ms - req.arrival_ms
            queue_delays.append(queue_delay)
            req.ttft_ms = queue_delay + batch_prefill_ms
            ttft_values.append(req.ttft_ms)
            class_totals[req.request_class]["ttft"].append(req.ttft_ms)

        while any(tokens > 0 for tokens in remaining_decode.values()):
            active = [req for req in admitted if remaining_decode[req.request_id] > 0]
            step_ms = _decode_step_ms(config.model, config.cache, cache, len(active))
            current_time_ms += step_ms
            active_page_samples.append(float(cache.resident_pages()))

            for req in active:
                remaining_decode[req.request_id] -= 1
                request_decode_elapsed[req.request_id] += step_ms
                throughput_tokens += 1

        for req in admitted:
            req.decode_latency_ms_per_token = request_decode_elapsed[req.request_id] / req.decode_tokens
            decode_latency_values.append(req.decode_latency_ms_per_token)
            class_totals[req.request_class]["decode"].append(req.decode_latency_ms_per_token)
            req.completed = True
            completed_requests += 1
            cache.release_request(req.request_id)

    makespan_s = max(current_time_ms / 1000.0, 1e-6)
    metrics = {
        "ttft_ms": _summary(ttft_values),
        "decode_latency_ms_per_token": _summary(decode_latency_values),
        "throughput_tokens_per_s": throughput_tokens / makespan_s,
        "peak_kv_memory_mb": cache.peak_memory_mb(),
        "average_active_kv_pages": (
            sum(active_page_samples) / len(active_page_samples) if active_page_samples else 0.0
        ),
        "cache_hit_rate": cache.snapshot()["cache_hit_rate"],
        "prefix_reuse_rate": cache.snapshot()["prefix_reuse_rate"],
        "evictions": cache.metrics["evictions"],
        "oom_avoided": cache.metrics["oom_avoided"],
        "oom_failures": cache.metrics["oom_failures"],
        "fragmentation_ratio": cache.average_fragmentation_ratio(),
        "queue_delay_ms": _summary(queue_delays),
        "completed_requests": completed_requests,
        "failed_requests": failed_requests,
    }

    per_class = {}
    for request_class, class_metrics in class_totals.items():
        per_class[request_class] = {
            "ttft_ms": _summary(class_metrics["ttft"]),
            "decode_latency_ms_per_token": _summary(class_metrics["decode"]),
        }

    trace_summary = {
        "total_requests": len(requests),
        "admitted_requests": completed_requests,
        "failed_requests": failed_requests,
        "burst_window_ms": config.workload.burst_window_ms,
    }

    notes = [
        "Hot prefixes represent system prompts, common templates, and agent tool schemas.",
        "Request-aware policy protects interactive traffic first, long-context second, and batch analytics last.",
        "Cost-aware mode quantizes or evicts low-value reusable pages when the KV budget crosses the high watermark.",
    ]

    return RuntimeResult(
        artifact_kind="kv_cache_serving_bench",
        config=config.to_dict(),
        metrics=metrics,
        per_class=per_class,
        scheduler=scheduler.snapshot(),
        cache=cache.snapshot(),
        trace_summary=trace_summary,
        experiment_notes=notes,
    )


def _shrink_batch_to_fit(
    batch: List[RequestSpec],
    cache: PagedKVCache,
    model: ModelConfig,
    cache_cfg: CacheConfig,
) -> List[RequestSpec]:
    if not batch:
        return batch

    available_capacity = max(0, cache.budget_bytes - cache.used_capacity_bytes)
    admitted: List[RequestSpec] = []
    running_capacity = 0

    for req in batch:
        estimate = estimate_request_capacity_bytes(model, cache_cfg, req)
        if not admitted:
            admitted.append(req)
            running_capacity += estimate
            continue
        if running_capacity + estimate <= available_capacity:
            admitted.append(req)
            running_capacity += estimate

    return admitted or batch[:1]


def estimate_request_capacity_bytes(
    model: ModelConfig,
    cache_cfg: CacheConfig,
    request: RequestSpec,
) -> int:
    total_tokens = request.total_prompt_tokens + request.decode_tokens
    kv_bytes = total_tokens * model.kv_bytes_per_token(cache_cfg.default_kv_mode)
    page_overhead = (
        math.ceil(total_tokens / cache_cfg.block_size_tokens)
        * cache_cfg.page_table_overhead_bytes
    )
    activation_window = request.dynamic_prompt_tokens
    if cache_cfg.chunked_prefill_tokens > 0:
        activation_window = min(activation_window, cache_cfg.chunked_prefill_tokens)
    transient_bytes = activation_window * model.activation_bytes_per_token
    return kv_bytes + page_overhead + transient_bytes


def _materialize_request(
    cache: PagedKVCache,
    config: ServingConfig,
    request: RequestSpec,
    now_ms: float,
) -> bool:
    if not cache.allocate_request_pages(
        request_id=request.request_id,
        request_class=request.request_class,
        total_tokens=request.dynamic_prompt_tokens + request.decode_tokens,
        now_ms=now_ms,
    ):
        return False

    for prefix in request.prefixes:
        blocks = math.ceil(prefix.tokens / config.cache.block_size_tokens)
        for block_index in range(blocks):
            token_count = min(
                config.cache.block_size_tokens,
                prefix.tokens - block_index * config.cache.block_size_tokens,
            )
            cache.ensure_prefix_block(
                prefix=prefix,
                block_index=block_index,
                token_count=token_count,
                request_class=request.request_class,
                now_ms=now_ms,
            )
    return True


def _prefix_accounting(cache: PagedKVCache, cache_cfg: CacheConfig, request: RequestSpec) -> tuple[int, int]:
    hit_tokens = 0
    miss_tokens = 0
    for prefix in request.prefixes:
        blocks = math.ceil(prefix.tokens / cache_cfg.block_size_tokens)
        for block_index in range(blocks):
            token_count = min(
                cache_cfg.block_size_tokens,
                prefix.tokens - block_index * cache_cfg.block_size_tokens,
            )
            block_key = f"{prefix.cache_key}:blk:{block_index}"
            page = cache.pages.get(block_key)
            if page is not None and page.access_count > 1:
                hit_tokens += token_count
            else:
                miss_tokens += token_count
    return hit_tokens, miss_tokens


def _prefill_latency_ms(
    model: ModelConfig,
    cache_cfg: CacheConfig,
    request: RequestSpec,
    prefix_hit_tokens: int,
    prefix_miss_tokens: int,
) -> float:
    dynamic_tokens = request.dynamic_prompt_tokens
    prefill_ms = (
        prefix_hit_tokens * model.prefill_hit_ms_per_token
        + prefix_miss_tokens * model.prefill_miss_ms_per_token
        + dynamic_tokens * model.dynamic_prefill_ms_per_token
    )
    if request.request_class == "long_context":
        prefill_ms *= 1.18
        if cache_cfg.chunked_prefill_tokens <= 0 and dynamic_tokens > 4096:
            prefill_ms *= 1.22
    if cache_cfg.chunked_prefill_tokens > 0 and dynamic_tokens > cache_cfg.chunked_prefill_tokens:
        chunks = math.ceil(dynamic_tokens / cache_cfg.chunked_prefill_tokens)
        efficiency = 0.80 + min(0.10, cache_cfg.chunked_prefill_tokens / 4096.0)
        prefill_ms = prefill_ms * efficiency + chunks * 1.1
    return prefill_ms


def _decode_step_ms(
    model: ModelConfig,
    cache_cfg: CacheConfig,
    cache: PagedKVCache,
    active_requests: int,
) -> float:
    resident_pages = cache.resident_pages()
    step_ms = (
        model.decode_base_ms
        + active_requests * model.decode_ms_per_request
        + resident_pages * model.decode_page_penalty_ms
    )
    if cache_cfg.block_size_tokens < 16:
        step_ms += (16 - cache_cfg.block_size_tokens) * model.page_table_penalty_ms
    elif cache_cfg.block_size_tokens > 16 and cache_cfg.block_size_tokens <= 32:
        step_ms += 0.06
    elif cache_cfg.block_size_tokens > 32:
        step_ms += 0.38 + 0.004 * (cache_cfg.block_size_tokens - 32)

    if cache.capacity_ratio() >= cache_cfg.high_watermark:
        step_ms *= 0.97
        if cache_cfg.cost_aware_mode:
            step_ms += 0.06

    if cache_cfg.default_kv_mode in {"fp8", "int8"}:
        step_ms = step_ms * 0.96 + 0.04

    return step_ms


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "p50": _percentile(sorted_values, 0.50),
        "p95": _percentile(sorted_values, 0.95),
        "mean": sum(sorted_values) / len(sorted_values),
        "max": sorted_values[-1],
    }


def _percentile(sorted_values: List[float], pct: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = min(len(sorted_values) - 1, max(0, int(round(pct * (len(sorted_values) - 1)))))
    return sorted_values[idx]
