/**
 * @file jacobi_cuda.cuh
 * @brief Declaração das implementações do método de Jacobi com CUDA.
 */

#ifndef JACOBI_CUDA_CUH
#define JACOBI_CUDA_CUH

/**
 * @brief Executa o método de Jacobi no GPU (versão naive com memória global).
 * 
 * Cada thread calcula um ponto da malha diretamente da memória global.
 * Testa múltiplas geometrias de bloco (8x8, 16x16, 32x8, 32x16, 32x32).
 * Inclui medição de tempo GPU com cudaEvent.
 * 
 * @param h_A_in      Matriz de entrada no host.
 * @param h_A_out     Matriz de saída no host.
 * @param N           Dimensão da matriz (N x N).
 * @param iterations  Número de iterações do método iterativo.
 * @param gpu_time_ms Tempo de execução na GPU em milissegundos (output).
 * @param block_x     Largura do bloco CUDA (default 16).
 * @param block_y     Altura do bloco CUDA (default 16).
 */
void run_jacobi_cuda_naive(const float* h_A_in, float* h_A_out, int N, int iterations, float& gpu_time_ms, int block_x = 16, int block_y = 16);

/**
 * @brief Executa o método de Jacobi no GPU com Shared Memory otimizada.
 * 
 * Utiliza tile compartilhado com padding para eliminar bank conflicts.
 * Carregamento colaborativo contíguo sem divergência de warp.
 * Configuração fixa: bloco 32x8 com tile 34x10 (padding incluso).
 * 
 * @param h_A_in      Matriz de entrada no host.
 * @param h_A_out     Matriz de saída no host.
 * @param N           Dimensão da matriz (N x N).
 * @param iterations  Número de iterações do método iterativo.
 * @param gpu_time_ms Tempo de execução na GPU em milissegundos (output).
 * @param block_x     Largura do bloco CUDA (default 32).
 * @param block_y     Altura do bloco CUDA (default 8).
 */
void run_jacobi_cuda_shared(const float* h_A_in, float* h_A_out, int N, int iterations, float& gpu_time_ms, int block_x = 32, int block_y = 8);

#endif