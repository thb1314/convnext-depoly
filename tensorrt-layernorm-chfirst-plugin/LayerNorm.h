#ifndef LAYER_NORM_H
#define LAYER_NORM_H

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
                            int opt_version);

#endif