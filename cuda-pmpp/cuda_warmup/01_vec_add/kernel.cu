#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vec_add_kernel(const float *a, const float *b, float *c, int n)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

torch::Tensor vec_add(torch::Tensor a, torch::Tensor b)
{
    const int n = a.numel();
    const int threadsPerBlock = 256;
    const int nBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    const auto c = torch::empty_like(a);
    vec_add_kernel<<<nBlocks, threadsPerBlock>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vec_add", &vec_add, "vec_add (CUDA)");
}
