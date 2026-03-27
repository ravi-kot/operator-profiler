from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence

from kv_cache_lab.cache import PagedKVCache
from kv_cache_lab.config import CacheConfig, ModelConfig
from kv_cache_lab.types import RequestSpec


class ReuseAwareScheduler:
    def __init__(self, model: ModelConfig, cache_cfg: CacheConfig, policy: str) -> None:
        self.model = model
        self.cache_cfg = cache_cfg
        self.policy = policy
        self.stats: Dict[str, float] = {
            "batches": 0.0,
            "reuse_priority_batches": 0.0,
            "reduced_batch_events": 0.0,
        }

    def plan_batch(self, ready: Sequence[RequestSpec], cache: PagedKVCache) -> List[RequestSpec]:
        if not ready:
            return []

        limit = self._effective_limit(cache)
        self.stats["batches"] += 1.0

        if self.policy == "fifo":
            return list(ready[:limit])

        groups: Dict[tuple, List[RequestSpec]] = defaultdict(list)
        for req in ready:
            groups[req.shared_prefix_key].append(req)

        ranked_groups = sorted(
            groups.values(),
            key=lambda group: self._group_score(group),
            reverse=True,
        )

        batch: List[RequestSpec] = []
        for group in ranked_groups:
            for req in sorted(group, key=lambda item: (-item.request_weight, item.arrival_ms)):
                batch.append(req)
                if len(batch) >= limit:
                    self.stats["reuse_priority_batches"] += 1.0
                    return batch

        return batch[:limit]

    def _effective_limit(self, cache: PagedKVCache) -> int:
        limit = self.cache_cfg.max_batch_size
        if self.cache_cfg.cost_aware_mode and cache.capacity_ratio() >= self.cache_cfg.high_watermark:
            limit = max(1, math.ceil(limit / 2))
            self.stats["reduced_batch_events"] += 1.0
        return limit

    def _group_score(self, group: Sequence[RequestSpec]) -> float:
        shared_tokens = sum(prefix.tokens for prefix in group[0].prefixes)
        priority_bonus = sum(req.request_weight for req in group)
        size_bonus = len(group) * 4.0
        return shared_tokens * len(group) + priority_bonus + size_bonus

    def snapshot(self) -> Dict[str, float]:
        return dict(self.stats)
