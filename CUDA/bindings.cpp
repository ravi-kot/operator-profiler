#include <torch/extension.h>

std::vector<torch::Tensor> quantize_pages_int8_cuda(torch::Tensor input);
torch::Tensor dequantize_pages_int8_cuda(
    torch::Tensor quantized,
    torch::Tensor scale,
    std::vector<int64_t> shape
);

std::vector<torch::Tensor> quantize_pages_int8(torch::Tensor input) {
  return quantize_pages_int8_cuda(input);
}

torch::Tensor dequantize_pages_int8(
    torch::Tensor quantized,
    torch::Tensor scale,
    std::vector<int64_t> shape
) {
  return dequantize_pages_int8_cuda(quantized, scale, shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_pages_int8", &quantize_pages_int8, "Row-wise int8 quantization for KV pages");
  m.def("dequantize_pages_int8", &dequantize_pages_int8, "Row-wise dequantization for KV pages");
}
