/**
 * @file jacobi_omp.h
 * @brief Declaração das implementações paralelas do método de Jacobi com OpenMP.
 */

#ifndef JACOBI_OMP_H
#define JACOBI_OMP_H

/**
 * @brief Resolve a equação de Laplace 2D com OpenMP (versão básica).
 * 
 * Paraleliza o loop externo da malha usando #pragma omp for com schedule estático.
 * Inclui #pragma omp simd para vetorização automática do loop interno.
 * Ponteiros são trocados dentro de uma região #pragma omp single para evitar race conditions.
 * 
 * @param A_in   Matriz de entrada com condições de contorno.
 * @param A_out  Matriz de saída com o resultado final.
 * @param N      Dimensão da matriz (N x N).
 * @param iterations Número de iterações do método iterativo.
 */
void jacobi_omp_basic(const float* A_in, float* A_out, int N, int iterations);

/**
 * @brief Resolve a equação de Laplace 2D com OpenMP usando tiling (blocos).
 * 
 * Particiona a malha em blocos (tiles) para melhorar a localidade de cache.
 * Cada thread/processador trabalha sobre um bloco da matriz, reduzindo contensões.
 * 
 * @param A_in      Matriz de entrada com condições de contorno.
 * @param A_out     Matriz de saída com o resultado final.
 * @param N         Dimensão da matriz (N x N).
 * @param iterations Número de iterações do método iterativo.
 * @param tile_size Tamanho de cada bloco (tile) na partição 2D.
 */
void jacobi_omp_tiled(const float* A_in, float* A_out, int N, int iterations, int tile_size);

#endif