from __future__ import annotations

import random
from typing import List

from kv_cache_lab.config import WorkloadConfig
from kv_cache_lab.types import PrefixSpec, RequestSpec


def build_prefix_catalog() -> List[PrefixSpec]:
    return [
        PrefixSpec(
            name="assistant-system-v2",
            tokens=256,
            namespace="system",
            pinned_candidate=True,
            popularity=0.98,
        ),
        PrefixSpec(
            name="chat-template-tools",
            tokens=192,
            namespace="template",
            pinned_candidate=True,
            popularity=0.92,
        ),
        PrefixSpec(
            name="agent-tool-schema",
            tokens=448,
            namespace="schema",
            pinned_candidate=True,
            popularity=0.88,
        ),
        PrefixSpec(
            name="analytics-template",
            tokens=288,
            namespace="template",
            pinned_candidate=False,
            popularity=0.54,
        ),
        PrefixSpec(
            name="research-template",
            tokens=640,
            namespace="template",
            pinned_candidate=False,
            popularity=0.36,
        ),
    ]


def _pick_request_class(rng: random.Random, cfg: WorkloadConfig) -> str:
    x = rng.random()
    if x < cfg.interactive_share:
        return "interactive"
    if x < cfg.interactive_share + cfg.batch_share:
        return "batch_analytics"
    return "long_context"


def _sample_dynamic_tokens(rng: random.Random, request_class: str) -> int:
    if request_class == "interactive":
        return rng.randint(120, 420)
    if request_class == "batch_analytics":
        return rng.randint(380, 1400)
    return rng.randint(3200, 9800)


def _sample_decode_tokens(rng: random.Random, request_class: str) -> int:
    if request_class == "interactive":
        return rng.randint(96, 240)
    if request_class == "batch_analytics":
        return rng.randint(80, 180)
    return rng.randint(128, 320)


def _prefixes_for_request(
    rng: random.Random,
    request_class: str,
    catalog: List[PrefixSpec],
) -> List[PrefixSpec]:
    system = catalog[0]
    tools = catalog[1]
    schema = catalog[2]
    analytics = catalog[3]
    research = catalog[4]

    if request_class == "interactive":
        prefixes = [system, tools]
        if rng.random() < 0.84:
            prefixes.append(schema)
        return prefixes

    if request_class == "batch_analytics":
        prefixes = [system, analytics]
        if rng.random() < 0.40:
            prefixes.append(schema)
        return prefixes

    prefixes = [system, research]
    if rng.random() < 0.55:
        prefixes.append(schema)
    return prefixes


def generate_burst_requests(cfg: WorkloadConfig) -> List[RequestSpec]:
    rng = random.Random(cfg.seed)
    catalog = build_prefix_catalog()
    requests: List[RequestSpec] = []

    for idx in range(cfg.total_requests):
        request_class = _pick_request_class(rng, cfg)
        prefixes = _prefixes_for_request(rng, request_class, catalog)
        arrival_ms = float(rng.randint(0, cfg.burst_window_ms))
        requests.append(
            RequestSpec(
                request_id=f"req-{idx:04d}",
                arrival_ms=arrival_ms,
                request_class=request_class,
                prefixes=prefixes,
                dynamic_prompt_tokens=_sample_dynamic_tokens(rng, request_class),
                decode_tokens=_sample_decode_tokens(rng, request_class),
                conversation_id=f"conv-{rng.randint(0, max(4, cfg.concurrency // 3)):03d}",
            )
        )

    requests.sort(key=lambda req: (req.arrival_ms, req.request_id))
    return requests
