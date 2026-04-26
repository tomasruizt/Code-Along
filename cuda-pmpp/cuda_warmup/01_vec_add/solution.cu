#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vec_add_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

torch::Tensor vec_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(a.numel() == b.numel(), "size mismatch");

    auto c = torch::empty_like(a);
    int n = a.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                                        n);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vec_add", &vec_add, "vec_add (CUDA)");
}
