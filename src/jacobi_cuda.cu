/**
 * @file jacobi_cuda.cu
 * @brief Implementações CUDA do método de Jacobi para solução da equação de Laplace 2D.
 * 
 * Inclui duas versões:
 * 1. Versão ingênua (naive): cada thread acessa diretamente a memória global.
 * 2. Versão otimizada com Shared Memory: tiles compartilhados com padding para eliminar bank conflicts.
 */

#include "jacobi_cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

/**
 * @brief Kernel CUDA naive — memória global sem cache intermediário.
 * 
 * Cada thread calcula um ponto da malha diretamente:
 *   A_new[i][j] = 0.25 × (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])
 * 
 * Mapeamento: blockIdx.x → colunas (j), blockIdx.y → linhas (i).
 * Verifica limites para não acessar fora da malha (i, j ∈ [1, N-2]).
 * 
 * @param A      Matriz de entrada na memória do dispositivo (const).
 * @param A_new  Matriz de saída na memória do dispositivo.
 * @param N      Dimensão da matriz (N x N).
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

/**
 * @brief Kernel CUDA otimizado com Shared Memory e padding anti-bank-conflict.
 * 
 * Estratégia de otimização:
 * 1. **Tiling**: Cada bloco carrega um tile (34×10) na shared memory antes de processar.
 * 2. **Padding**: Array [TILE_Y][TILE_X+1] = [10][35] elimina bank conflicts no CUDA.
 * 3. **Carregamento colaborativo**: threads do bloco carregam contiguamente sem divergência.
 * 4. **Guards**: max(0, min(N-1, coord)) lida com bordas da malha naturalmente.
 * 
 * Configuração fixa: bloco 32×8 threads, tile 34×10 (com padding de 1 coluna).
 * 
 * @param A      Matriz de entrada na memória do dispositivo (const).
 * @param A_new  Matriz de saída na memória do dispositivo.
 * @param N      Dimensão da matriz (N x N).
 */
#define BLOCK_X 32
#define BLOCK_Y 8
#define TILE_X (BLOCK_X + 2)
#define TILE_Y (BLOCK_Y + 2)
#define TILE_SIZE (TILE_X * TILE_Y)

__global__ void jacobi_cuda_shared_optimized_kernel(const float* __restrict__ A, 
                                                     float* __restrict__ A_new, 
                                                     int N) {
    // PADDING: [TILE_Y][TILE_X + 1] -> 35 colunas (elimina 100% dos bank conflicts)
    __shared__ float tile[TILE_Y][TILE_X + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    int gx_start = blockIdx.x * blockDim.x - 1;
    int gy_start = blockIdx.y * blockDim.y - 1;

    // Carregamento colaborativo contíguo sem divergência de warp
    for (int i = tid; i < TILE_SIZE; i += threads_per_block) {
        int local_y = i / TILE_X;
        int local_x = i % TILE_X;

        int gx = gx_start + local_x;
        int gy = gy_start + local_y;

        gx = max(0, min(N - 1, gx));
        gy = max(0, min(N - 1, gy));

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
 * @brief Executa o método de Jacobi no GPU (versão ingênua com memória global).
 * 
 * Fluxo:
 * 1. Aloca memória no dispositivo (cudaMalloc) e copia dados host→device.
 * 2. Configura grid/blocks baseado nas dimensões fornecidas.
 * 3. Executa kernel em loop com ping-pong buffering (swap de ponteiros device).
 * 4. Mede tempo com cudaEvent (alta precisão, sincronizado).
 * 5. Copia resultado device→host e libera recursos.
 * 
 * @param h_A_in      Matriz de entrada no host.
 * @param h_A_out     Matriz de saída no host.
 * @param N           Dimensão da matriz (N x N).
 * @param iterations  Número de iterações do método iterativo.
 * @param gpu_time_ms Tempo de execução na GPU em milissegundos (output).
 * @param block_x     Largura do bloco CUDA (default 16).
 * @param block_y     Altura do bloco CUDA (default 16).
 */
void run_jacobi_cuda_naive(const float* h_A_in, float* h_A_out, int N, int iterations, float& gpu_time_ms, int block_x, int block_y) {
    size_t bytes = N * N * sizeof(float);
    float *d_A, *d_A_new;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_A_new, bytes);

    cudaMemcpy(d_A, h_A_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_new, h_A_in, bytes, cudaMemcpyHostToDevice);

    dim3 threads(block_x, block_y);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mede tempo end-to-end: Host→Device + Kernels + Device→Host
    cudaMemcpy(d_A, h_A_in, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);

    float* d_src = d_A;
    float* d_dst = d_A_new;
    for (int iter = 0; iter < iterations; ++iter) {
        jacobi_cuda_naive_kernel<<<blocks, threads>>>(d_src, d_dst, N);
        std::swap(d_src, d_dst);
    }

    cudaMemcpy(h_A_out, d_src, bytes, cudaMemcpyDeviceToHost);
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
 * @brief Executa o método de Jacobi no GPU com Shared Memory otimizada.
 * 
 * Diferenças da versão naive:
 * - Kernel usa shared memory com padding anti-bank-conflict ([10][35] ao invés de [10][34]).
 * - Carregamento colaborativo contíguo sem divergência de warp.
 * - Bloco fixo 32×8 para casar com TILE_X=34 e TILE_Y=10.
 * 
 * @param h_A_in      Matriz de entrada no host.
 * @param h_A_out     Matriz de saída no host.
 * @param N           Dimensão da matriz (N x N).
 * @param iterations  Número de iterações do método iterativo.
 * @param gpu_time_ms Tempo de execução na GPU em milissegundos (output).
 * @param block_x     Largura do bloco CUDA (default 32, forçado para 32 no kernel).
 * @param block_y     Altura do bloco CUDA (default 8, forçado para 8 no kernel).
 */
void run_jacobi_cuda_shared(const float* h_A_in, float* h_A_out, int N, int iterations, float& gpu_time_ms, int block_x, int block_y) {
    size_t bytes = N * N * sizeof(float);
    float *d_A, *d_A_new;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_A_new, bytes);

    cudaMemcpy(d_A, h_A_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_new, h_A_in, bytes, cudaMemcpyHostToDevice);

    // Força 32x8 para casar com as constantes TILE_X e TILE_Y do kernel shared
    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mede tempo end-to-end: Host→Device + Kernels + Device→Host
    cudaMemcpy(d_A, h_A_in, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);

    float* d_src = d_A;
    float* d_dst = d_A_new;
    for (int iter = 0; iter < iterations; ++iter) {
        jacobi_cuda_shared_optimized_kernel<<<blocks, threads>>>(d_src, d_dst, N);
        std::swap(d_src, d_dst);
    }

    cudaMemcpy(h_A_out, d_src, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(h_A_out, d_src, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_A_new);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}