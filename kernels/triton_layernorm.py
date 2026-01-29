"""
    Triton LayerNorm forward kernel — normalized over last dimension, optional affine.
    Matches PyTorch F.layer_norm(x, normalized_shape, weight, bias, eps) for correctness checks.
    On Windows Triton is not available on PyPI; we fall back to F.layer_norm so the benchmark still runs.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _layernorm_fwd_kernel(
        X,
        Y,
        W,
        B,
        stride_row,
        N,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        """Forward LayerNorm over last dim: y = (x - mean) / sqrt(var + eps) * w + b."""
        row = tl.program_id(0)
        X_row = X + row * stride_row
        Y_row = Y + row * stride_row

        # First pass: compute mean (vector accumulation then sum, like Triton tutorial)
        _sum = tl.zeros([BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)
            a = tl.load(X_row + cols, mask=cols < N, other=0.0).to(tl.float32)
            _sum += a
        mean = tl.sum(_sum, axis=0) / N

        # Second pass: compute variance
        _var = tl.zeros([BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)
            x = tl.load(X_row + cols, mask=cols < N, other=0.0).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.0)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)

        # Third pass: normalize and affine
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)
            mask = cols < N
            x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0)
            b = tl.load(B + cols, mask=mask, other=0.0)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            tl.store(Y_row + cols, y, mask=mask)


def triton_layernorm(
    x: torch.Tensor,
    normalized_shape: tuple,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    LayerNorm over the last len(normalized_shape) dimensions.
    x: (..., *normalized_shape), weight/bias: (normalized_shape).
    Uses Triton kernel when available (Linux); on Windows falls back to F.layer_norm.
    """
    if not TRITON_AVAILABLE:
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    if x.dim() < 1 or list(normalized_shape) != list(x.shape[-len(normalized_shape) :]):
        raise ValueError("normalized_shape must match the last dimensions of x")
    x_flat = x.reshape(-1, x.shape[-1])
    M, N = x_flat.shape

    y = torch.empty_like(x_flat, dtype=x.dtype, device=x.device)

    max_block = 65536 // x.element_size()
    BLOCK_N = min(triton.next_power_of_2(N), max_block, 4096)

    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M,)
    _layernorm_fwd_kernel[grid](
        x_flat,
        y,
        weight,
        bias,
        stride_row=x_flat.stride(0),
        N=N,
        eps=eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return y.reshape_as(x)
