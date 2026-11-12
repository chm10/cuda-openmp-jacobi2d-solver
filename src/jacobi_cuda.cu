/**
 * @file jacobi_cuda.cu
 * @brief Implementações CUDA do método de Jacobi para solução da equação de Laplace 2D.
 * 
 * Inclui duas versões:
 * 1. Versão direta (naive): cada thread acessa diretamente a memória global.
 * 2. Versão com Shared Memory: tiles compartilhados (34x10) com padding horizontal de 1 coluna.
 */

#include "jacobi_cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

/**
 * @brief Kernel CUDA naive — leitura direta da memória global.
 */
__global__ void jacobi_cuda_naive_kernel(const float* __restrict__ A, 
                                         float* __restrict__ A_new, 
                                         int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        A_new[i * N + j] = 0.25f * (A[(i - 1) * N + j] +
                                    A[(i + 1) * N + j] +
                                    A[i * N + (j - 1)] +
                                    A[i * N + (j + 1)]);
    }
}

#define BLOCK_X 32
#define BLOCK_Y 8
#define TILE_X (BLOCK_X + 2)
#define TILE_Y (BLOCK_Y + 2)
#define TILE_SIZE (TILE_X * TILE_Y)

/**
 * @brief Kernel CUDA com Shared Memory e padding horizontal de 1 coluna ([10][35]).
 */
__global__ void jacobi_cuda_shared_optimized_kernel(const float* __restrict__ A, 
                                                     float* __restrict__ A_new, 
                                                     int N) {
    // Padding de 1 coluna para evitar conflitos de banco na leitura do estêncil
    __shared__ float tile[TILE_Y][TILE_X + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    int gx_start = blockIdx.x * blockDim.x - 1;
    int gy_start = blockIdx.y * blockDim.y - 1;

    // Carregamento colaborativo do bloco e halo
    for (int i = tid; i < TILE_SIZE; i += threads_per_block) {
        int local_y = i / TILE_X;
        int local_x = i % TILE_X;

        int gx = max(0, min(N - 1, gx_start + local_x));
        int gy = max(0, min(N - 1, gy_start + local_y));

        tile[local_y][local_x] = A[gy * N + gx];
    }

    __syncthreads();

    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    if (gy > 0 && gy < N - 1 && gx > 0 && gx < N - 1) {
        int sm_x = tx + 1;
        int sm_y = ty + 1;

        A_new[gy * N + gx] = 0.25f * (tile[sm_y - 1][sm_x] +
                                      tile[sm_y + 1][sm_x] +
                                      tile[sm_y][sm_x - 1] +
                                      tile[sm_y][sm_x + 1]);
    }
}

/**
 * @brief Executa o método de Jacobi na GPU (versão naive) medindo o tempo dos kernels.
 */
void run_jacobi_cuda_naive(const float* h_A_in, float* h_A_out, int N, int iterations, float& gpu_time_ms, int block_x, int block_y) {
    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    float *d_A = nullptr, *d_A_new = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_A_new, bytes);

    cudaMemcpy(d_A, h_A_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_new, h_A_in, bytes, cudaMemcpyHostToDevice);

    dim3 threads(block_x, block_y);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    float* d_src = d_A;
    float* d_dst = d_A_new;
    for (int iter = 0; iter < iterations; ++iter) {
        jacobi_cuda_naive_kernel<<<blocks, threads>>>(d_src, d_dst, N);
        std::swap(d_src, d_dst);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(h_A_out, d_src, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_A_new);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * @brief Executa o método de Jacobi na GPU (versão Shared Memory 32x8) medindo o tempo dos kernels.
 */
void run_jacobi_cuda_shared(const float* h_A_in, float* h_A_out, int N, int iterations, float& gpu_time_ms, int block_x, int block_y) {
    (void)block_x;
    (void)block_y;

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    float *d_A = nullptr, *d_A_new = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_A_new, bytes);

    cudaMemcpy(d_A, h_A_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_new, h_A_in, bytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    float* d_src = d_A;
    float* d_dst = d_A_new;
    for (int iter = 0; iter < iterations; ++iter) {
        jacobi_cuda_shared_optimized_kernel<<<blocks, threads>>>(d_src, d_dst, N);
        std::swap(d_src, d_dst);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(h_A_out, d_src, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_A_new);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}