#include <cuda.h>

#include "gelu.h"
#include "checkMacrosPlugin.h"

namespace GELU
{

constexpr float A = 1.41421356237309504;  // sqrt(2)

template <typename T>
__device__ inline T myerf(const T& x);

template <>
__device__ inline float myerf(const float& x)
{
    return erf(x);
}

template <>
__device__ inline half myerf(const half& x)
{
    const float tmp = erf(__half2float(x));
    return __float2half(tmp);
}

template <>
__device__ inline half2 myerf(const half2& x)
{
    // at the moment, there is no half2 tanh builtin
    float2 tmp = (__half22float2(x));
    tmp.x = erf(tmp.x);
    tmp.y = erf(tmp.y);
    return __float22half2_rn(tmp);
}



template <typename T, unsigned TPB>
__global__ void geluKernel(const T a, int n, const T* input, T* output)
{
    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n)
    {
        const T in = input[idx];
        // const T cdf = erf(in * (T)(0.5f) * a);
        const T cdf = (T)0.5f * ((T)1.0f + myerf(in * (T)0.5f * a));
        output[idx] = in * cdf;
    }
}

template <unsigned TPB>
__global__ void geluKernel(const half2 a, int n, const half2* input, half2* output, const half2 b, const half2 c)
{
    const int idx = blockIdx.x * TPB + threadIdx.x;
    // b = __floats2half2_rn(0.5f, 0.5f)
    // c = __floats2half2_rn(1.0f, 1.0f)
    if (idx < n)
    {
        const half2 in = input[idx];
        const half2 cdf = b * (c + myerf(in * b * a));
        output[idx] = in * cdf;
    }
}

__device__ inline half2 tanh(const half2& x)
{
    // at the moment, there is no half2 tanh builtin
    float2 tmp = (__half22float2(x));
    tmp.x = tanhf(tmp.x);
    tmp.y = tanhf(tmp.y);
    return __float22half2_rn(tmp);
}

// constants for approximating the normal cdf
constexpr float TRT_A = 0.5f;
constexpr float TRT_B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
constexpr float TRT_C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)

template <typename T, unsigned TPB>
__global__ void trt_geluKernel(const T a, const T b, const T c, int n, const T* input, T* output)
{
    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n)
    {
        const T in = input[idx];
        const T cdf = a + a * tanh(in * (c * in * in + b));
        output[idx] = in * cdf;
    }
}

int computeGelu(cudaStream_t stream, int n, const float* input, float* output)
{
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    geluKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(A, n, input, output);

    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

int computeGelu(cudaStream_t stream, int n, const half* input, half* output)
{
    constexpr int blockSize = 256;
    
    const int gridSize = (n + blockSize - 1) / blockSize;
    geluKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(A, n, input, output);

    CHECK_CUDA(cudaPeekAtLastError());
    return 0;
}

} // namespace bert

