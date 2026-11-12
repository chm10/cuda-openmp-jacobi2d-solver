/**
 * @file jacobi_seq.h
 * @brief Declaração da implementação sequencial do método de Jacobi.
 */

#ifndef JACOBI_SEQ_H
#define JACOBI_SEQ_H

/**
 * @brief Resolve a equação de Laplace 2D usando o método iterativo de Jacobi (versão sequencial).
 * 
 * Utiliza duas matrizes (bufA e bufB) com alternância (ping-pong) entre iterações.
 * Cada ponto interno é atualizado como a média dos 4 vizinhos da matriz corrente.
 * 
 * @param A_in   Matriz de entrada com condições de contorno.
 * @param A_out  Matriz de saída com o resultado final.
 * @param N      Dimensão da matriz (N x N).
 * @param iterations Número de iterações do método iterativo.
 */
void jacobi_seq(const float* A_in, float* A_out, int N, int iterations);

#endif