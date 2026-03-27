# KV Cache Serving Lab

`KV Cache Serving Lab` is an inference-systems project for studying how KV-cache policy affects LLM serving performance under realistic multi-request workloads.

The project started from an operator-profiling codebase and was refactored into a serving-focused lab with:

- paged KV cache simulation
- hot-prefix pinning
- request-aware cache policy
- reuse-aware scheduling
- cost-aware memory control
- CUDA extension scaffolding for KV-page operations

The goal is not just to benchmark one model run. The goal is to measure how cache layout, prefix reuse, scheduling, and memory pressure interact when many requests compete for the same GPU budget.

## Overview

Modern LLM serving is heavily shaped by KV-cache behavior. Long contexts, repeated system prompts, tool schemas, and bursty request traffic can all turn KV cache into a first-order latency and memory bottleneck.

This repository focuses on the serving layer around that problem:

- which prefixes should be pinned
- which requests should get priority
- how batching should exploit shared prefixes
- how memory policy should react when capacity becomes tight
- how to evaluate these decisions with system-level metrics

## Core Features

### Paged KV Cache

The runtime models KV storage as fixed-size token blocks so cache usage can be reasoned about in terms of pages, reuse, and fragmentation.

### Hot-Prefix Pinning

Frequently reused prefixes such as system prompts, templates, and tool schemas can be pinned so they remain resident across requests.

### Request-Aware Policy

The cache policy distinguishes between:

- interactive chat
- batch analytics
- long-context jobs

This lets the runtime preserve low-latency behavior for high-priority traffic while still supporting heavier workloads.

### Reuse-Aware Scheduler

The scheduler forms batches by prefix overlap instead of pure FIFO order, increasing cache reuse and improving throughput under bursty traffic.

### Cost-Aware Mode

When memory pressure rises, the runtime can respond by:

- quantizing lower-value KV pages
- evicting stale reusable blocks
- reducing effective batch size

## Metrics Tracked

The project tracks serving metrics that matter at the systems level:

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

The latest local benchmark summary from `artifacts/summary.json` reports:

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

- At 100 concurrent requests, the reuse-aware serving path reduced TTFT P50 from `10964 ms` to `6988 ms`.
- At 100 concurrent requests, throughput improved from `1613 tok/s` to `1817 tok/s` versus the baseline path.
- KV quantization reduced peak KV memory by about `50%` versus fp16 in the synthetic study.

## Repository Structure

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

You can open [dashboard/index.html](c:/Users/Admin/Workspace/operator-profiler/dashboard/index.html) directly, or run a local server:

```powershell
cd c:\Users\Admin\Workspace\operator-profiler\dashboard
python -m http.server 8080 --bind 127.0.0.1
```

Then visit `http://127.0.0.1:8080`.

## Main Commands

Run the primary benchmark:

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

Run the KV-quantization study:

```powershell
python -m bench.experiment_kv_quantization --out artifacts\kv_quantization.json
```

Rebuild the consolidated summary:

```powershell
python -m bench.summarize --out-dir artifacts
```

## Experiments Included

### Burst Load

Compares a baseline FIFO path against the reuse-aware serving path at `10`, `50`, and `100` concurrent requests.

### Block-Size Sweep

Sweeps `8`, `16`, `32`, and `64` token blocks to study the tradeoff between reuse and fragmentation.

### Chunked Prefill

Measures how chunking affects TTFT and long-context stability.

### KV Quantization

Compares fp16, fp8-style, and int8 KV storage modes for memory savings, latency, and reconstruction fidelity.

## CUDA Support

The [CUDA](c:/Users/Admin/Workspace/operator-profiler/CUDA) directory contains the extension scaffold for KV-page operations:

- `bindings.cpp`
- `kv_page_ops.cu`
- `setup.py`

This path is intended for future live GPU integration such as:

- KV-page quantization and dequantization
- page-table compaction
- prefix hashing or lookup acceleration
- allocator helpers for paged KV blocks

## Notes

- The current local environment reported `torch 2.10.0+cpu`, so the latest run used CPU fallback rather than a live CUDA execution path.
- The runtime is currently a synthetic serving simulator designed to isolate cache-policy behavior without requiring model downloads.
- The target hardware configuration in the project settings is an RTX 5070-class machine with an 8 GB VRAM budget.

## Roadmap

Planned improvements:

- connect the runtime to a real model-serving path
- run the full suite on the target CUDA machine
- wire the CUDA extension into live quantize/dequantize calls
- add allocator and page-table microbenchmarks
- extend the scheduler to multi-GPU or disaggregated-cache scenarios
- evaluate quality impact of quantized KV modes with a real model
