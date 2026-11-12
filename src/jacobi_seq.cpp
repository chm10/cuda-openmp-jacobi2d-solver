/**
 * @file jacobi_seq.cpp
 * @brief Implementação sequencial do método de Jacobi para solução da equação de Laplace 2D.
 */

#include "jacobi_seq.h"
#include <vector>
#include <algorithm>

/**
 * @brief Resolve a equação de Laplace 2D com o método iterativo de Jacobi (sequencial).
 * 
 * Algoritmo:
 * 1. Copia a matriz de entrada para dois buffers (ping-pong buffering).
 * 2. Para cada iteração, atualiza cada ponto interno como média dos 4 vizinhos.
 * 3. Alterna os ponteiros curr/next após cada iteração (swap).
 * 4. Copia o resultado final para A_out.
 * 
 * Complexidade: O(N² × iterations) — sem paralelismo.
 * 
 * @param A_in   Matriz de entrada com condições de contorno fixas.
 * @param A_out  Matriz de saída com a solução convergida.
 * @param N      Dimensão da matriz (N x N).
 * @param iterations Número de iterações do método iterativo.
 */
void jacobi_seq(const float* A_in, float* A_out, int N, int iterations) {
    std::vector<float> bufA(A_in, A_in + (N * N));
    std::vector<float> bufB(A_in, A_in + (N * N));

    float* curr = bufA.data();
    float* next = bufB.data();

    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                next[i * N + j] = 0.25f * (curr[(i - 1) * N + j] +
                                          curr[(i + 1) * N + j] +
                                          curr[i * N + (j - 1)] +
                                          curr[i * N + (j + 1)]);
            }
        }
        std::swap(curr, next);
    }

    std::copy(curr, curr + (N * N), A_out);
}