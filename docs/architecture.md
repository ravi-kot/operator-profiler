# KV Cache Serving Lab

This project reframes the original operator benchmark into an inference-systems lab.

Core differentiators:

- Hot-prefix pinning for system prompts, common templates, and repeated tool schemas.
- Request-aware cache policy that treats interactive chat, batch analytics, and long-context jobs differently.
- Reuse-aware scheduler that batches by shared-prefix overlap instead of simple FIFO.
- Cost-aware mode that quantizes low-value KV pages, evicts stale reusable blocks, and shrinks batch size under pressure.

Tracked metrics:

- TTFT
- Decode latency per token
- Throughput in tokens per second
- Peak KV memory
- Average active KV pages
- Cache hit rate
- Prefix reuse rate
- Evictions
- OOM avoided
- Fragmentation ratio

Experiment set:

- Burst load at 10, 50, and 100 concurrent requests.
- Block-size sweep at 8, 16, 32, and 64 tokens.
- Chunked prefill sensitivity for long-context jobs.
- KV quantization tradeoff study.

Target hardware stance:

- The simulator is tuned for an 88 GB VRAM workstation budget.
- The current environment can run the Python fallback path.
- The `CUDA/` folder is the extension point for custom kernels on a CUDA-enabled setup.
