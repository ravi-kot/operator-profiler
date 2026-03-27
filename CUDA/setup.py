from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="kv_cache_cuda",
    ext_modules=[
        CUDAExtension(
            name="kv_cache_cuda",
            sources=["bindings.cpp", "kv_page_ops.cu"],
            extra_compile_args={
                "cxx": ["/O2"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
