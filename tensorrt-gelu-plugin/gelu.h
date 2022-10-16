#ifndef GELU_H
#define GELU_H

#include <cuda_fp16.h>

namespace GELU
{
    int computeGelu(cudaStream_t stream, int n, const float* input, float* output);
    int computeGelu(cudaStream_t stream, int n, const half* input, half* output);
}
#endif
