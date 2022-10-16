#include <cub/cub.cuh>
#include "LayerNorm.h"

#define FINAL_MASK 0xffffffff

template<typename T>
inline __device__ T hmul2(T a, T b) {
    return __hmul2(a, b);
}

template<typename T>
inline __device__ T hsub2(T a, T b) {
    return __hsub2(a, b);
}

template<typename T>
inline __device__ T hadd2(T a, T b) {
    return __hadd2(a, b);
}

template<typename T>
struct TypeConverter {using Type = half2;}; // keep for generality

template<>
struct TypeConverter<half2> {using Type = half;};

template<>
struct TypeConverter<half> {using Type = half2;};

template<typename T>
inline __device__ T ldg(const T* val) {
    return __ldg(val);
}

template<typename T>
inline __device__ T float2type(float a);

template<>
inline __device__ half float2type(float a) {
    return __float2half_rn(a);
}

template<typename T>
inline __device__ T float2type2(float a);

template<>
inline __device__ half2 float2type2(float a) {
    return __float2half2_rn(a);
}

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    }
    return val;
}

template<typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T)(0.0f);
}

template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template<typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val)
{
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T)0.0f;
}


template<typename T>
__global__ void generalLayerNorm(
    const T* __restrict input, const T* __restrict gamma, const T* __restrict beta, T* output, int m, int n, float epsilon)
{
    const int offset = blockIdx.x * n;

    // __shared__ T s_mean;
    // __shared__ T s_variance;
    T mean = 0.0f;
    T variance = 0.0f;
    /**
    b -> blockIdx.x gridDim.x
    c -> threadDim.x (blockDim.x, n)
    s -> blockIdx.y gridDim.y
    [blockIdx.x, tid...n , blockIdx.y]
    index = (blockIdx.x * n + threadDim.x) * gridDim.y + blockIdx.y
    
    index = (blockIdx.x * n + i) * gridDim.y + blockIdx.y
    
    **/
    T local_mean = (T)(0.0f);
    T local_var = (T)(0.0f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val = ldg(&input[(offset + i) * gridDim.y + blockIdx.y]);
        T tmp = val / (T)n;
        local_mean += tmp;
        local_var += tmp * val;
    }

    mean = blockReduceSum(local_mean);
    variance = blockReduceSum(local_var);

    // if (threadIdx.x == 0) {
    //     s_mean = mean;
    //     s_variance = rsqrtf(variance - s_mean * s_mean + (T)(epsilon));
    // }
    // __syncthreads();
    // s_mean = mean;
    variance = rsqrtf(variance - mean * mean + (T)(epsilon));
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T beta_val = (beta == nullptr) ? (T)(0.0f) : ldg(&beta[i]);
        int index = (offset + i) * gridDim.y + blockIdx.y;
        output[index] = ((input[index] - mean) * variance) * ldg(&gamma[i]) + beta_val;
    }
}

template<typename T>
void invokeGeneralLayerNorm(T* out,
                            const T* input,
                            const T* gamma,
                            const T* beta,
                            const int m,
                            const int n,
                            const int last_dim,
                            float epsilon,
                            cudaStream_t stream,
                            int opt_version)
// last_dim: 对[m / last_dim, n, last_dim] 数据进行LayerNorm，且在n这个维度
{
    dim3 grid(m / last_dim, last_dim);

    dim3 block(min(n, 1024));
    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0) {
        block.x = (block.x + 31) / 32 * 32;
    }

    /* should pay attention to the rsqrt precision*/
    generalLayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, beta, out, m, n, epsilon);  // For gpt-3

}

template void invokeGeneralLayerNorm<float>(float* out,
                            const float* input,
                            const float* gamma,
                            const float* beta,
                            const int m,
                            const int n,
                            const int last_dim,
                            float epsilon,
                            cudaStream_t stream,
                            int opt_version);

template void invokeGeneralLayerNorm<half>(half* out,
                            const half* input,
                            const half* gamma,
                            const half* beta,
                            const int m,
                            const int n,
                            const int last_dim,
                            float epsilon,
                            cudaStream_t stream,
                            int opt_version);
                            
