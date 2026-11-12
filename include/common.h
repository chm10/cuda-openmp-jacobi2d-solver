/**
 * @file common.h
 * @brief Funções auxiliares comuns ao benchmark Jacobi 2D.
 */

#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

/**
 * @brief Inicializa a matriz com condições de contorno para o método de Jacobi.
 * 
 * As bordas são configuradas da seguinte forma:
 * - Topo e base: 100.0 (temperatura fixa)
 * - Esquerda e direita: 0.0 (temperatura fixa)
 * - Interior: 0.0 (condição inicial)
 * 
 * @param grid Ponteiro para a matriz linearizada (tamanho N*N).
 * @param N Dimensão da matriz (N x N).
 */
inline void init_matrix(float* grid, int N) {
    std::fill(grid, grid + (N * N), 0.0f);
    for (int i = 0; i < N; ++i) {
        grid[0 * N + i] = 100.0f;           // Borda Superior
        grid[(N - 1) * N + i] = 100.0f;     // Borda Inferior
        grid[i * N + 0] = 0.0f;             // Borda Esquerda
        grid[i * N + (N - 1)] = 0.0f;       // Borda Direita
    }
}

#endif