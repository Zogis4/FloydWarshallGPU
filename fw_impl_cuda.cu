// fw_impl_cuda.cu
#include <cuda_runtime.h>
#include "fw_common.h"

__global__ void floydWarshallKernel(int *dist, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        int ik = i * n + k;
        int kj = k * n + j;
        int ij = i * n + j;
        int dik = dist[ik];
        int kdj = dist[kj];
        int dij = dist[ij];
        if (dik != INF && kdj != INF) {
            int cand = dik + kdj;
            if (cand < dij) dist[ij] = cand;
        }
    }
}

void floydWarshallCUDA(int **dist, int n) {
    size_t bytes = (size_t)n * (size_t)n * sizeof(int);
    int *h = (int*)malloc(bytes);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h[i*n + j] = dist[i][j];

    int *d = NULL;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

    dim3 bloco(16, 16);
    dim3 grade((n + bloco.x - 1) / bloco.x,
               (n + bloco.y - 1) / bloco.y);

    for (int k = 0; k < n; k++) {
        floydWarshallKernel<<<grade, bloco>>>(d, n, k);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dist[i][j] = h[i*n + j];

    cudaFree(d);
    free(h);
}
