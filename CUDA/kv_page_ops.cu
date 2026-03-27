#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

__global__ void rowwise_maxabs_kernel(
    const float* input,
    float* scale,
    int rows,
    int cols
) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float local_max = 0.0f;
  for (int idx = threadIdx.x; idx < cols; idx += blockDim.x) {
    float value = fabsf(input[row * cols + idx]);
    local_max = fmaxf(local_max, value);
  }

  __shared__ float buffer[256];
  buffer[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scale[row] = fmaxf(buffer[0] / 127.0f, 1e-6f);
  }
}

__global__ void quantize_kernel(
    const float* input,
    const float* scale,
    int8_t* output,
    int rows,
    int cols
) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (index >= total) {
    return;
  }
  int row = index / cols;
  float scaled = input[index] / scale[row];
  float clipped = fminf(127.0f, fmaxf(-127.0f, nearbyintf(scaled)));
  output[index] = static_cast<int8_t>(clipped);
}

__global__ void dequantize_kernel(
    const int8_t* input,
    const float* scale,
    float* output,
    int rows,
    int cols
) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (index >= total) {
    return;
  }
  int row = index / cols;
  output[index] = static_cast<float>(input[index]) * scale[row];
}

}  // namespace

std::vector<torch::Tensor> quantize_pages_int8_cuda(torch::Tensor input) {
  auto contiguous = input.contiguous().to(torch::kFloat32);
  auto rows = contiguous.size(0);
  auto cols = contiguous.numel() / rows;
  auto flattened = contiguous.view({rows, cols});

  auto scale = torch::zeros({rows}, contiguous.options().dtype(torch::kFloat32));
  auto output = torch::zeros({rows, cols}, contiguous.options().dtype(torch::kInt8));

  rowwise_maxabs_kernel<<<rows, 256, 0, at::cuda::getDefaultCUDAStream()>>>(
      flattened.data_ptr<float>(),
      scale.data_ptr<float>(),
      rows,
      cols
  );

  int total = rows * cols;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  quantize_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      flattened.data_ptr<float>(),
      scale.data_ptr<float>(),
      output.data_ptr<int8_t>(),
      rows,
      cols
  );

  return {output, scale};
}

torch::Tensor dequantize_pages_int8_cuda(
    torch::Tensor quantized,
    torch::Tensor scale,
    std::vector<int64_t> shape
) {
  auto contiguous = quantized.contiguous();
  auto rows = contiguous.size(0);
  auto cols = contiguous.numel() / rows;
  auto output = torch::zeros({rows, cols}, scale.options().dtype(torch::kFloat32));

  int total = rows * cols;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  dequantize_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      contiguous.data_ptr<int8_t>(),
      scale.contiguous().data_ptr<float>(),
      output.data_ptr<float>(),
      rows,
      cols
  );

  return output.view(shape);
}
