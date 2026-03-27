# CUDA Extension

This folder holds the custom CUDA extension for the KV cache project.


- `bindings.cpp`: PyTorch extension bindings.
- `kv_page_ops.cu`: CUDA kernels for row-wise int8 KV page quantization and dequantization.
- `setup.py`: build entry point for `python setup.py install`.



- It gives you a clean place to extend page compaction, prefix hashing, or page-table ops later.
- The Python runtime already has a fallback path, so the benchmark suite still runs without compiling this extension.

Next CUDA additions:

- Page-table compaction kernels.
- Prefix hash matching on-device.
- FP8 packing once the target stack is fixed.
