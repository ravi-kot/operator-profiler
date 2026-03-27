from __future__ import annotations

from typing import Dict

import torch

try:
    import kv_cache_cuda  # type: ignore
except ImportError:
    kv_cache_cuda = None


def extension_status() -> Dict[str, object]:
    return {
        "extension_available": kv_cache_cuda is not None,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def quantize_tensor(x: torch.Tensor, mode: str) -> Dict[str, torch.Tensor]:
    if mode == "fp16":
        return {
            "quantized": x.to(torch.float16),
            "scale": torch.ones(x.shape[0], device=x.device, dtype=torch.float32),
        }

    if (
        kv_cache_cuda is not None
        and x.is_cuda
        and mode == "int8"
        and hasattr(kv_cache_cuda, "quantize_pages_int8")
    ):
        quantized, scale = kv_cache_cuda.quantize_pages_int8(x.contiguous())
        return {"quantized": quantized, "scale": scale}

    row_view = x.reshape(x.shape[0], -1).float()
    scale = row_view.abs().amax(dim=1).clamp_min(1e-6) / 127.0
    quantized = torch.round(row_view / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return {"quantized": quantized, "scale": scale}


def dequantize_tensor(quantized: torch.Tensor, scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if (
        kv_cache_cuda is not None
        and quantized.is_cuda
        and hasattr(kv_cache_cuda, "dequantize_pages_int8")
    ):
        return kv_cache_cuda.dequantize_pages_int8(quantized.contiguous(), scale.contiguous(), list(shape))
    dequant = quantized.float() * scale.unsqueeze(1)
    return dequant.reshape(shape)
