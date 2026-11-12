/**
 * @file validation.h
 * @brief Declaração da função de validação de resultados.
 */

#ifndef VALIDATION_H
#define VALIDATION_H

/**
 * @brief Estrutura contendo os erros máximo absoluto e relativo entre duas matrizes.
 */
struct ValidationResult {
    double max_abs_error;  ///< Erro máximo absoluto: |ref[i] - test[i]|
    double max_rel_error;  ///< Erro máximo relativo: |ref[i] - test[i]| / (|ref[i]| + eps)
};

/**
 * @brief Compara dois arrays e calcula os erros máximo absoluto e relativo.
 * 
 * Utiliza epsilon = 1e-9 para evitar divisão por zero no erro relativo.
 * 
 * @param ref  Matriz de referência (ground truth).
 * @param test Matriz a ser validada contra a referência.
 * @param N    Dimensão da matriz (N x N).
 * @return ValidationResult Contendo os erros calculados.
 */
ValidationResult validate_results(const float* ref, const float* test, int N);

#endif