from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


REQUEST_CLASS_WEIGHTS = {
    "interactive": 3.0,
    "long_context": 2.0,
    "batch_analytics": 1.0,
}


@dataclass(frozen=True)
class PrefixSpec:
    name: str
    tokens: int
    namespace: str
    pinned_candidate: bool
    popularity: float

    @property
    def cache_key(self) -> str:
        return f"{self.namespace}:{self.name}"


@dataclass
class RequestSpec:
    request_id: str
    arrival_ms: float
    request_class: str
    prefixes: List[PrefixSpec]
    dynamic_prompt_tokens: int
    decode_tokens: int
    conversation_id: str

    ttft_ms: float = 0.0
    decode_latency_ms_per_token: float = 0.0
    completed: bool = False
    failed: bool = False
    failure_reason: Optional[str] = None

    @property
    def shared_prefix_key(self) -> Tuple[str, ...]:
        return tuple(prefix.cache_key for prefix in self.prefixes)

    @property
    def total_prefix_tokens(self) -> int:
        return sum(prefix.tokens for prefix in self.prefixes)

    @property
    def total_prompt_tokens(self) -> int:
        return self.total_prefix_tokens + self.dynamic_prompt_tokens

    @property
    def request_weight(self) -> float:
        return REQUEST_CLASS_WEIGHTS[self.request_class]


@dataclass
class PageState:
    page_id: int
    block_key: str
    resident_tokens: int
    capacity_tokens: int
    bytes_used: int
    bytes_capacity: int
    storage_mode: str
    reusable: bool
    pinned: bool
    request_class: str
    request_id: Optional[str]
    last_touch_ms: float
    access_count: int = 0
    prefix_key: Optional[str] = None

    def touch(self, now_ms: float) -> None:
        self.last_touch_ms = now_ms
        self.access_count += 1


@dataclass
class RuntimeResult:
    artifact_kind: str
    config: dict
    metrics: dict
    per_class: dict
    scheduler: dict
    cache: dict
    trace_summary: dict
    experiment_notes: List[str] = field(default_factory=list)
