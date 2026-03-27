from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from kv_cache_lab.config import CacheConfig, ModelConfig
from kv_cache_lab.types import PageState, PrefixSpec, REQUEST_CLASS_WEIGHTS


class PagedKVCache:
    def __init__(self, model: ModelConfig, cache_cfg: CacheConfig) -> None:
        self.model = model
        self.cache_cfg = cache_cfg
        self.budget_bytes = int(model.kv_budget_gb * (1024 ** 3))
        self.pages: Dict[str, PageState] = {}
        self.request_pages: Dict[str, List[str]] = defaultdict(list)
        self.next_page_id = 1
        self.used_bytes = 0
        self.used_capacity_bytes = 0
        self.peak_bytes = 0
        self.metrics = {
            "prefix_lookups": 0,
            "prefix_hits": 0,
            "reusable_prefix_lookups": 0,
            "reusable_prefix_hits": 0,
            "evictions": 0,
            "quantized_pages": 0,
            "quantization_events": 0,
            "oom_avoided": 0,
            "oom_failures": 0,
            "pinned_pages": 0,
        }
        self.fragmentation_samples: List[float] = []

    def capacity_ratio(self) -> float:
        if self.budget_bytes <= 0:
            return 1.0
        return self.used_capacity_bytes / self.budget_bytes

    def resident_pages(self) -> int:
        return len(self.pages)

    def current_memory_mb(self) -> float:
        return self.used_bytes / (1024 ** 2)

    def peak_memory_mb(self) -> float:
        return self.peak_bytes / (1024 ** 2)

    def page_bytes(self, token_count: int, mode: str) -> Tuple[int, int]:
        bytes_per_token = self.model.kv_bytes_per_token(mode)
        used_bytes = token_count * bytes_per_token
        capacity_bytes = (
            self.cache_cfg.block_size_tokens * bytes_per_token
            + self.cache_cfg.page_table_overhead_bytes
        )
        return used_bytes, capacity_bytes

    def pin_hot_prefixes(self, prefixes: Iterable[PrefixSpec], now_ms: float) -> None:
        if not self.cache_cfg.hot_prefix_pinning:
            return
        for prefix in prefixes:
            if not prefix.pinned_candidate:
                continue
            for block_key, token_count in self._split_prefix(prefix):
                if block_key in self.pages:
                    continue
                self._allocate_page(
                    block_key=block_key,
                    token_count=token_count,
                    storage_mode=self.cache_cfg.default_kv_mode,
                    request_class="interactive",
                    request_id=None,
                    reusable=True,
                    pinned=True,
                    prefix_key=prefix.cache_key,
                    now_ms=now_ms,
                )
                self.metrics["pinned_pages"] += 1

    def ensure_prefix_block(
        self,
        prefix: PrefixSpec,
        block_index: int,
        token_count: int,
        request_class: str,
        now_ms: float,
    ) -> bool:
        block_key = f"{prefix.cache_key}:blk:{block_index}"
        self.metrics["prefix_lookups"] += 1
        self.metrics["reusable_prefix_lookups"] += 1

        page = self.pages.get(block_key)
        if page is not None:
            page.touch(now_ms)
            self.metrics["prefix_hits"] += 1
            self.metrics["reusable_prefix_hits"] += 1
            return True

        if not self._ensure_capacity_for(
            token_count, self.cache_cfg.default_kv_mode, request_class, now_ms
        ):
            self.metrics["oom_failures"] += 1
            return False

        self._allocate_page(
            block_key=block_key,
            token_count=token_count,
            storage_mode=self.cache_cfg.default_kv_mode,
            request_class=request_class,
            request_id=None,
            reusable=True,
            pinned=False,
            prefix_key=prefix.cache_key,
            now_ms=now_ms,
        )
        return False

    def allocate_request_pages(
        self,
        request_id: str,
        request_class: str,
        total_tokens: int,
        now_ms: float,
    ) -> bool:
        blocks = math.ceil(total_tokens / self.cache_cfg.block_size_tokens)
        for block_idx in range(blocks):
            resident_tokens = min(
                self.cache_cfg.block_size_tokens,
                total_tokens - block_idx * self.cache_cfg.block_size_tokens,
            )
            if resident_tokens <= 0:
                continue
            if not self._ensure_capacity_for(
                resident_tokens,
                self.cache_cfg.default_kv_mode,
                request_class,
                now_ms,
            ):
                self.release_request(request_id)
                self.metrics["oom_failures"] += 1
                return False
            block_key = f"{request_id}:blk:{block_idx}"
            self._allocate_page(
                block_key=block_key,
                token_count=resident_tokens,
                storage_mode=self.cache_cfg.default_kv_mode,
                request_class=request_class,
                request_id=request_id,
                reusable=False,
                pinned=False,
                prefix_key=None,
                now_ms=now_ms,
            )
        return True

    def release_request(self, request_id: str) -> None:
        for block_key in self.request_pages.pop(request_id, []):
            page = self.pages.pop(block_key, None)
            if page is None:
                continue
            self.used_bytes -= page.bytes_used
            self.used_capacity_bytes -= page.bytes_capacity
        self._record_fragmentation()

    def snapshot(self) -> Dict[str, float]:
        prefix_lookups = self.metrics["prefix_lookups"]
        reusable_lookups = self.metrics["reusable_prefix_lookups"]
        return {
            "resident_pages": float(self.resident_pages()),
            "current_memory_mb": self.current_memory_mb(),
            "peak_memory_mb": self.peak_memory_mb(),
            "cache_hit_rate": (
                self.metrics["prefix_hits"] / prefix_lookups if prefix_lookups else 0.0
            ),
            "prefix_reuse_rate": (
                self.metrics["reusable_prefix_hits"] / reusable_lookups
                if reusable_lookups
                else 0.0
            ),
            "evictions": float(self.metrics["evictions"]),
            "oom_avoided": float(self.metrics["oom_avoided"]),
            "oom_failures": float(self.metrics["oom_failures"]),
            "fragmentation_ratio": self.average_fragmentation_ratio(),
            "pinned_pages": float(self.metrics["pinned_pages"]),
        }

    def average_fragmentation_ratio(self) -> float:
        if not self.fragmentation_samples:
            return 0.0
        return sum(self.fragmentation_samples) / len(self.fragmentation_samples)

    def _split_prefix(self, prefix: PrefixSpec) -> Iterable[Tuple[str, int]]:
        blocks = math.ceil(prefix.tokens / self.cache_cfg.block_size_tokens)
        for block_index in range(blocks):
            token_count = min(
                self.cache_cfg.block_size_tokens,
                prefix.tokens - block_index * self.cache_cfg.block_size_tokens,
            )
            yield f"{prefix.cache_key}:blk:{block_index}", token_count

    def _ensure_capacity_for(
        self,
        token_count: int,
        storage_mode: str,
        request_class: str,
        now_ms: float,
    ) -> bool:
        _, capacity_bytes = self.page_bytes(token_count, storage_mode)
        if self.used_capacity_bytes + capacity_bytes <= self.budget_bytes:
            return True

        saved = False
        if self.cache_cfg.cost_aware_mode:
            saved = self._cost_aware_relief(capacity_bytes, request_class, now_ms)
            if saved:
                self.metrics["oom_avoided"] += 1
        return saved

    def _cost_aware_relief(
        self,
        required_capacity_bytes: int,
        request_class: str,
        now_ms: float,
    ) -> bool:
        if self.used_capacity_bytes + required_capacity_bytes <= self.budget_bytes:
            return True

        if self.cache_cfg.emergency_kv_mode != self.cache_cfg.default_kv_mode:
            self._quantize_victims(request_class, now_ms)
            if self.used_capacity_bytes + required_capacity_bytes <= self.budget_bytes:
                return True

        self._evict_victims(request_class, now_ms, required_capacity_bytes)
        return self.used_capacity_bytes + required_capacity_bytes <= self.budget_bytes

    def _quantize_victims(self, request_class: str, now_ms: float) -> None:
        victims = self._victim_pages(request_class)
        for page in victims:
            if page.pinned or page.storage_mode == self.cache_cfg.emergency_kv_mode:
                continue
            old_used = page.bytes_used
            old_capacity = page.bytes_capacity
            new_used, new_capacity = self.page_bytes(
                page.resident_tokens, self.cache_cfg.emergency_kv_mode
            )
            page.storage_mode = self.cache_cfg.emergency_kv_mode
            page.bytes_used = new_used
            page.bytes_capacity = new_capacity
            page.touch(now_ms)
            self.used_bytes += new_used - old_used
            self.used_capacity_bytes += new_capacity - old_capacity
            self.metrics["quantized_pages"] += 1
            self.metrics["quantization_events"] += 1
            self._update_peak()
            self._record_fragmentation()
            if self.capacity_ratio() <= self.cache_cfg.high_watermark:
                return

    def _evict_victims(
        self,
        request_class: str,
        now_ms: float,
        required_capacity_bytes: int,
    ) -> None:
        del now_ms
        for page in self._victim_pages(request_class):
            if page.pinned or not page.reusable:
                continue
            self.pages.pop(page.block_key, None)
            self.used_bytes -= page.bytes_used
            self.used_capacity_bytes -= page.bytes_capacity
            self.metrics["evictions"] += 1
            self._record_fragmentation()
            if self.used_capacity_bytes + required_capacity_bytes <= self.budget_bytes:
                return

    def _victim_pages(self, request_class: str) -> List[PageState]:
        incoming_weight = REQUEST_CLASS_WEIGHTS.get(request_class, 1.0)

        def score(page: PageState) -> float:
            priority = REQUEST_CLASS_WEIGHTS.get(page.request_class, 1.0)
            pin_bonus = 1000.0 if page.pinned else 0.0
            reuse_bonus = 8.0 if page.reusable else 0.0
            class_penalty = priority * 10.0
            recency_bonus = page.last_touch_ms / 1000.0
            access_bonus = page.access_count * 4.0
            incoming_penalty = 2.0 if priority >= incoming_weight else 0.0
            return (
                pin_bonus
                + reuse_bonus
                + class_penalty
                + recency_bonus
                + access_bonus
                + incoming_penalty
            )

        return sorted(self.pages.values(), key=score)

    def _allocate_page(
        self,
        block_key: str,
        token_count: int,
        storage_mode: str,
        request_class: str,
        request_id: Optional[str],
        reusable: bool,
        pinned: bool,
        prefix_key: Optional[str],
        now_ms: float,
    ) -> None:
        used_bytes, capacity_bytes = self.page_bytes(token_count, storage_mode)
        page = PageState(
            page_id=self.next_page_id,
            block_key=block_key,
            resident_tokens=token_count,
            capacity_tokens=self.cache_cfg.block_size_tokens,
            bytes_used=used_bytes,
            bytes_capacity=capacity_bytes,
            storage_mode=storage_mode,
            reusable=reusable,
            pinned=pinned,
            request_class=request_class,
            request_id=request_id,
            last_touch_ms=now_ms,
            access_count=1,
            prefix_key=prefix_key,
        )
        self.next_page_id += 1
        self.pages[block_key] = page
        self.used_bytes += used_bytes
        self.used_capacity_bytes += capacity_bytes
        if request_id is not None:
            self.request_pages[request_id].append(block_key)
        self._update_peak()
        self._record_fragmentation()

    def _record_fragmentation(self) -> None:
        total_capacity_tokens = sum(page.capacity_tokens for page in self.pages.values())
        if total_capacity_tokens <= 0:
            self.fragmentation_samples.append(0.0)
            return
        used_tokens = sum(page.resident_tokens for page in self.pages.values())
        waste = max(0, total_capacity_tokens - used_tokens)
        self.fragmentation_samples.append(waste / total_capacity_tokens)

    def _update_peak(self) -> None:
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
