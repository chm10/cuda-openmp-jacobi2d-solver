/**
 * @file validation.cpp
 * @brief Implementação da validação de resultados entre implementações.
 */

#include "validation.h"
#include <cmath>
#include <algorithm>

/**
 * @brief Compara dois arrays e calcula os erros máximo absoluto e relativo.
 * 
 * Algoritmo:
 * 1. Percorre todos N² elementos comparando ref[i] com test[i].
 * 2. Erro absoluto: diff = |ref[i] - test[i]|
 * 3. Erro relativo: rel = diff / (|ref[i]| + eps) para evitar divisão por zero.
 * 4. Retorna os máximos encontrados em ValidationResult.
 * 
 * @param ref  Matriz de referência (ground truth — geralmente a versão sequencial).
 * @param test Matriz a ser validada contra a referência.
 * @param N    Dimensão da matriz (N x N).
 * @return ValidationResult Contendo max_abs_error e max_rel_error.
 */
ValidationResult validate_results(const float* ref, const float* test, int N) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    const double eps = 1e-9;

    for (int i = 0; i < N * N; ++i) {
        double diff = std::abs((double)ref[i] - (double)test[i]);
        double ref_val = std::abs((double)ref[i]);

        if (diff > max_abs) {
            max_abs = diff;
        }

        double rel = diff / (ref_val + eps);
        if (rel > max_rel) {
            max_rel = rel;
        }
    }

    return {max_abs, max_rel};
}