from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


SCALAR_NBYTES: Dict[str, int] = {
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
}


def scalar_nbytes(mode: str) -> int:
    if mode not in SCALAR_NBYTES:
        raise ValueError(f"Unsupported KV mode: {mode}")
    return SCALAR_NBYTES[mode]


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "Synthetic-13B-Inference-Serving-Lab"
    num_layers: int = 40
    num_kv_heads: int = 8
    head_dim: int = 128
    total_vram_gb: float = 8.0
    kv_budget_gb: float = 28.0
    activation_budget_gb: float = 6.0
    target_gpu: str = "RTX 5070-class workstation target with 8 GB total VRAM budget"
    prefill_miss_ms_per_token: float = 0.020
    prefill_hit_ms_per_token: float = 0.002
    dynamic_prefill_ms_per_token: float = 0.016
    decode_base_ms: float = 0.58
    decode_ms_per_request: float = 0.21
    decode_page_penalty_ms: float = 0.0012
    page_table_penalty_ms: float = 0.0035
    activation_bytes_per_token: int = 65536

    def kv_bytes_per_token(self, mode: str) -> int:
        return (
            self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * 2
            * scalar_nbytes(mode)
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CacheConfig:
    block_size_tokens: int = 16
    max_batch_size: int = 16
    hot_prefix_pinning: bool = True
    request_aware_policy: bool = True
    reuse_aware_scheduler: bool = True
    cost_aware_mode: bool = True
    default_kv_mode: str = "fp16"
    emergency_kv_mode: str = "fp8"
    high_watermark: float = 0.82
    critical_watermark: float = 0.92
    page_table_overhead_bytes: int = 12288
    chunked_prefill_tokens: int = 512
    keep_reusable_pages_after_decode: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadConfig:
    concurrency: int = 32
    total_requests: int = 96
    burst_window_ms: int = 250
    seed: int = 7
    interactive_share: float = 0.55
    batch_share: float = 0.25
    long_context_share: float = 0.20

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ServingConfig:
    name: str
    scheduler_policy: str
    cache_policy: str
    model: ModelConfig
    cache: CacheConfig
    workload: WorkloadConfig

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "scheduler_policy": self.scheduler_policy,
            "cache_policy": self.cache_policy,
            "model": self.model.to_dict(),
            "cache": self.cache.to_dict(),
            "workload": self.workload.to_dict(),
        }
