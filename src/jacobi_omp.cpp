/**
 * @file jacobi_omp.cpp
 * @brief Implementações paralelas do método de Jacobi com OpenMP (básico e com tiling).
 */

#include "jacobi_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

/**
 * @brief Resolve a equação de Laplace 2D com OpenMP (versão básica paralela).
 * 
 * Estratégia de paralelismo:
 * - #pragma omp for: distribui linhas da malha entre threads (schedule estático).
 * - #pragma omp simd: instrução ao compilador para vetorizar o loop interno.
 * - #pragma omp single: região exclusiva para trocar ponteiros curr/next (barreira implícita).
 * 
 * A remoção de 'nowait' no loop for força barreira antes da troca de ponteiros,
 * evitando race conditions entre leitura (curr) e escrita (next).
 * 
 * @param A_in   Matriz de entrada com condições de contorno fixas.
 * @param A_out  Matriz de saída com a solução convergida.
 * @param N      Dimensão da matriz (N x N).
 * @param iterations Número de iterações do método iterativo.
 */
void jacobi_omp_basic(const float* A_in, float* A_out, int N, int iterations) {
    std::vector<float> bufA(A_in, A_in + (N * N));
    std::vector<float> bufB(A_in, A_in + (N * N));

    float* curr = bufA.data();
    float* next = bufB.data();

    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            // REMOVIDO 'nowait': força a barreira antes da troca de ponteiros
            #pragma omp for schedule(static)
            for (int i = 1; i < N - 1; ++i) {
                #pragma omp simd
                for (int j = 1; j < N - 1; ++j) {
                    next[i * N + j] = 0.25f * (curr[(i - 1) * N + j] +
                                              curr[(i + 1) * N + j] +
                                              curr[i * N + (j - 1)] +
                                              curr[i * N + (j + 1)]);
                }
            }

            // O single troca os ponteiros e sua barreira garante que ninguém avança antes
            #pragma omp single
            {
                std::swap(curr, next);
            }
        }
    }

    std::copy(curr, curr + (N * N), A_out);
}

/**
 * @brief Resolve a equação de Laplace 2D com OpenMP usando tiling (partição em blocos).
 * 
 * Estratégia de tiling:
 * - A malha é particionada em blocos quadrados de tile_size × tile_size.
 * - Cada thread processa um bloco completo, melhorando a localidade de cache L1/L2.
 * - #pragma omp for distribui os blocos entre threads.
 * 
 * Vantagem sobre a versão básica: reduz acessos redundantes à memória global
 * ao reutilizar dados carregados no cache para todo o bloco.
 * 
 * @param A_in      Matriz de entrada com condições de contorno fixas.
 * @param A_out     Matriz de saída com a solução convergida.
 * @param N         Dimensão da matriz (N x N).
 * @param iterations Número de iterações do método iterativo.
 * @param tile_size Tamanho de cada bloco na partição 2D (ex: 32 = blocos de 32×32).
 */
void jacobi_omp_tiled(const float* A_in, float* A_out, int N, int iterations, int tile_size) {
    std::vector<float> bufA(A_in, A_in + (N * N));
    std::vector<float> bufB(A_in, A_in + (N * N));

    float* curr = bufA.data();
    float* next = bufB.data();

    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            // RESTAURADO: Particionamento 2D por blocos (Tiling)
            #pragma omp for schedule(static)
            for (int bi = 1; bi < N - 1; bi += tile_size) {
                int i_end = std::min(bi + tile_size, N - 1);
                for (int bj = 1; bj < N - 1; bj += tile_size) {
                    int j_end = std::min(bj + tile_size, N - 1);

                    for (int i = bi; i < i_end; ++i) {
                        #pragma omp simd
                        for (int j = bj; j < j_end; ++j) {
                            next[i * N + j] = 0.25f * (curr[(i - 1) * N + j] +
                                                      curr[(i + 1) * N + j] +
                                                      curr[i * N + (j - 1)] +
                                                      curr[i * N + (j + 1)]);
                        }
                    }
                }
            }

            #pragma omp single
            {
                std::swap(curr, next);
            }
        }
    }

    std::copy(curr, curr + (N * N), A_out);
}