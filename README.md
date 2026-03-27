# KV Cache Serving Lab

KV Cache Serving Lab is a systems-oriented project for evaluating how KV-cache policy shapes LLM serving behavior under shared-prefix, long-context, and burst-load workloads.

Instead of treating KV cache as a hidden implementation detail, this project makes it the main subject of study: how KV pages are allocated, reused, pinned, quantized, evicted, and scheduled when multiple requests compete for limited memory.

## Highlights

- Paged KV cache runtime with block-based accounting
- Hot-prefix pinning for system prompts, templates, and tool schemas
- Request-aware cache policy for interactive chat, batch analytics, and long-context jobs
- Reuse-aware scheduler that batches requests by prefix overlap
- Cost-aware mode that can quantize or evict lower-value cache state under pressure
- CUDA/C++ extension scaffold for future KV-page operations
- Experiment suite covering burst load, block-size sweeps, chunked prefill, and KV quantization
- Static dashboard and JSON/CSV artifact pipeline for result tracking

## Why This Project Exists

KV cache is one of the main bottlenecks in modern inference systems. As prompt lengths grow and request traffic becomes more bursty, serving performance depends not only on the model, but also on:

- how reusable prefixes are identified and kept resident
- how cache state is laid out and paged
- how requests are prioritized under memory pressure
- how batching changes when prefix reuse is available
- how memory savings affect latency and throughput

This repository focuses on that serving layer.

## Architecture

The project is built around four main ideas:

### 1. Paged KV Cache

The runtime models KV storage as fixed-size token blocks. This makes it possible to reason about:

- active KV pages
- memory footprint
- prefix reuse
- fragmentation
- eviction behavior

### 2. Hot-Prefix Pinning

High-frequency prefixes can be pinned so they remain resident across requests. The current workload model includes reusable prefixes such as:

- system prompts
- shared chat templates
- repeated tool schemas

### 3. Request-Aware Policy

The system treats different traffic classes differently:

- interactive chat
- batch analytics
- long-context jobs

This enables policy decisions that favor latency-sensitive traffic while still supporting larger batch-oriented workloads.

### 4. Reuse-Aware Scheduling

Instead of batching requests strictly by arrival order, the scheduler groups work by shared-prefix overlap. This increases prefix reuse and improves efficiency under bursty multi-request traffic.

### 5. Cost-Aware Memory Control

When the cache approaches its budget, the runtime can react by:

- quantizing lower-value KV pages
- evicting stale reusable pages
- shrinking effective batch size

## Metrics Tracked

The benchmark pipeline records the metrics that matter for serving systems:

- TTFT
- decode latency per token
- throughput in tokens/sec
- peak KV memory
- average active KV pages
- cache hit rate
- prefix reuse rate
- evictions
- OOM avoided
- fragmentation ratio

## Latest Results

The latest local summary in `artifacts/summary.json` reports:

| Metric | Value |
|---|---:|
| TTFT P50 | 2223.86 ms |
| Decode Latency / Token P50 | 4.74 ms |
| Throughput | 1901.55 tok/s |
| Peak KV Memory | 12460.78 MB |
| Average Active KV Pages | 1808.97 |
| Cache Hit Rate | 98.91% |
| Prefix Reuse Rate | 98.91% |
| Fragmentation Ratio | 0.0032 |

Selected experiment outcomes:

- At 100 concurrent requests, the reuse-aware path reduced TTFT P50 from `10964 ms` to `6988 ms`
- At 100 concurrent requests, throughput improved from `1613 tok/s` to `1817 tok/s`
- KV quantization reduced peak KV memory by about `50%` versus fp16 in the synthetic study

## Experiments

### Burst Load

Compares a baseline FIFO path against the reuse-aware path at:

- 10 concurrent requests
- 50 concurrent requests
- 100 concurrent requests

Goal:
- evaluate whether scheduling plus paged cache behavior remains stable as concurrency rises

### Block-Size Sweep

Sweeps:

- 8-token blocks
- 16-token blocks
- 32-token blocks
- 64-token blocks

Goal:
- study the tradeoff between prefix reuse and fragmentation

### Chunked Prefill

Tests chunked-prefill settings for long-context workloads.

Goal:
- measure TTFT and long-context stability under heavier prompt loads

### KV Quantization

Compares fp16, fp8-style, and int8 KV storage modes.

Goal:
- measure memory savings and observe any throughput or fidelity tradeoff

## Repository Layout

```text
operator-profiler/
|- bench/                 # benchmark entry points and experiment runners
|- kv_cache_lab/          # runtime, cache policy, scheduler, workload generation
|- CUDA/                  # CUDA/C++ extension scaffold for KV-page operations
|- dashboard/             # static dashboard for viewing benchmark results
|- docs/                  # architecture and project notes
|- artifacts/             # generated JSON/CSV outputs
|- kernels/               # legacy operator-kernel work retained from the original base
|- report/                # screenshots and supporting assets
```

## Getting Started

### Run the Full Suite

```powershell
cd c:\Users\Admin\Workspace\operator-profiler
python -m bench.run_all
```

This generates:

- `artifacts/kv_service.json`
- `artifacts/burst_load.json`
- `artifacts/block_sweep.json`
- `artifacts/chunked_prefill.json`
- `artifacts/kv_quantization.json`
- `artifacts/summary.json`
- `artifacts/summary.csv`
- `dashboard/summary.json`
- `dashboard/summary.js`

### Open the Dashboard

Open the dashboard directly or serve it locally:

```powershell
cd c:\Users\Admin\Workspace\operator-profiler\dashboard
python -m http.server 8080 --bind 127.0.0.1
```

Then visit `http://127.0.0.1:8080`.

## Main Commands

Run the main serving benchmark:

```powershell
python -m bench.kv_service_bench --out artifacts\kv_service.json
```

Run the burst-load experiment:

```powershell
python -m bench.experiment_burst_load --out artifacts\burst_load.json
```

Run the block-size sweep:

```powershell
python -m bench.experiment_block_sweep --out artifacts\block_sweep.json
```

Run the chunked-prefill study:

```powershell
python -m bench.experiment_chunked_prefill --out artifacts\chunked_prefill.json
```

Run the quantization study:

```powershell
python -m bench.experiment_kv_quantization --out artifacts\kv_quantization.json
```

Rebuild the consolidated summary:

```powershell
python -m bench.summarize --out-dir artifacts
```

## CUDA Support

The `CUDA/` directory contains the extension scaffold for future GPU-backed KV-page operations:

- `bindings.cpp`
- `kv_page_ops.cu`
- `setup.py`

Planned extension targets include:

- KV-page quantization and dequantization
- page-table compaction
- prefix lookup acceleration
- allocator helpers for paged KV blocks

## Notes

- The most recent local run used CPU fallback because the environment reported `torch 2.10.0+cpu`
- The current runtime is a synthetic serving simulator intended to isolate cache-policy behavior without requiring model downloads
- The target hardware configuration in the project settings is an RTX 5070-class machine with an 8 GB VRAM budget
- Reported memory metrics come from the simulator configuration and experiment model, not a live GPU memory trace

## Roadmap

Planned next steps:

- connect the runtime to a real model-serving path
- run the full suite on the target CUDA machine
- wire the CUDA extension into live quantize/dequantize calls
- add allocator and page-table microbenchmarks
- extend the scheduler to multi-GPU or disaggregated-cache scenarios
- evaluate quality impact of quantized KV modes with a real model
