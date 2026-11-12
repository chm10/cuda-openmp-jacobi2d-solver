/**
 * @file main.cpp
 * @brief Ponto de entrada do benchmark Jacobi 2D — executa e compara todas as implementações.
 * 
 * Fluxo principal:
 * 1. Parseia argumentos (N, iterations) ou usa defaults (4096, 1000).
 * 2. Inicializa matriz com condições de contorno.
 * 3. Executa 5 categorias de implementação:
 *    - CPU Sequencial (baseline)
 *    - OpenMP Basic (paralelismo de linha)
 *    - OpenMP Tiled (paralelismo com tiling para cache locality)
 *    - CUDA Naive (5 geometrias de bloco: 8x8, 16x16, 32x8, 32x16, 32x32)
 *    - CUDA Shared Memory (otimizado com tiles e padding anti-bank-conflict)
 * 4. Para cada versão: calcula tempo, speedup relativo ao sequencial, e valida resultados.
 */

#include "common.h"
#include "jacobi_seq.h"
#include "jacobi_omp.h"
#include "jacobi_cuda.cuh"
#include "validation.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

/**
 * @brief Função principal — executa o benchmark completo e imprime resultados.
 * 
 * Argumentos de linha de comando:
 *   argv[1] = N (dimensão da matriz, default 4096)
 *   argv[2] = iterations (número de iterações, default 1000)
 * 
 * @param argc Número de argumentos.
 * @param argv Argumentos de linha de comando.
 * @return 0 em caso de sucesso.
 */
int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int iterations = (argc > 2) ? std::atoi(argv[2]) : 1000;

    std::cout << "========================================================================================\n";
    std::cout << " Jacobi 2D Benchmark (N=" << N << ", iterations=" << iterations << ")\n";
    std::cout << "========================================================================================\n";

    std::vector<float> h_A(N * N);
    std::vector<float> h_seq(N * N);
    std::vector<float> h_omp_basic(N * N);
    std::vector<float> h_omp_tiled(N * N);
    std::vector<float> h_cuda_naive(N * N);
    std::vector<float> h_cuda_shared(N * N);

    init_matrix(h_A.data(), N);

    // 1. CPU Sequencial (Baseline)
    auto t0 = std::chrono::high_resolution_clock::now();
    jacobi_seq(h_A.data(), h_seq.data(), N, iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[CPU Seq]        Time: " << std::setw(8) << time_seq << " ms\n";

    // 2. OpenMP Basic
    t0 = std::chrono::high_resolution_clock::now();
    jacobi_omp_basic(h_A.data(), h_omp_basic.data(), N, iterations);
    t1 = std::chrono::high_resolution_clock::now();
    double time_omp = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto v_omp = validate_results(h_seq.data(), h_omp_basic.data(), N);
    std::cout << "[OpenMP Basic]   Time: " << std::setw(8) << time_omp << " ms | Speedup: " 
              << std::setw(6) << (time_seq / time_omp) << "x | Max Abs Err: " << v_omp.max_abs_error 
              << " | Max Rel Err: " << v_omp.max_rel_error << "\n";

    // 3. OpenMP Tiled
    t0 = std::chrono::high_resolution_clock::now();
    jacobi_omp_tiled(h_A.data(), h_omp_tiled.data(), N, iterations, 32);
    t1 = std::chrono::high_resolution_clock::now();
    double time_omp_tiled = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto v_omp_tiled = validate_results(h_seq.data(), h_omp_tiled.data(), N);
    std::cout << "[OpenMP Tiled]   Time: " << std::setw(8) << time_omp_tiled << " ms | Speedup: " 
              << std::setw(6) << (time_seq / time_omp_tiled) << "x | Max Abs Err: " << v_omp_tiled.max_abs_error 
              << " | Max Rel Err: " << v_omp_tiled.max_rel_error << "\n";

    // 4. CUDA Naive (Geometrias de Bloco)
    struct BlockConfig { int x; int y; };
    std::vector<BlockConfig> configs = {{8, 8}, {16, 16}, {32, 8}, {32, 16}, {32, 32}};

    for (const auto& cfg : configs) {
        float time_cuda_naive = 0.0f;
        run_jacobi_cuda_naive(h_A.data(), h_cuda_naive.data(), N, iterations, time_cuda_naive, cfg.x, cfg.y);
        auto v_cuda_n = validate_results(h_seq.data(), h_cuda_naive.data(), N);
        std::cout << "[CUDA Naive " << std::setw(2) << cfg.x << "x" << std::setw(2) << cfg.y << "] Time: " 
                  << std::setw(8) << time_cuda_naive << " ms | Speedup: " 
                  << std::setw(6) << (time_seq / time_cuda_naive) << "x | Max Abs Err: " << v_cuda_n.max_abs_error 
                  << " | Max Rel Err: " << v_cuda_n.max_rel_error << "\n";
    }

    // 5. CUDA Shared Memory Otimizado
    float time_cuda_shared = 0.0f;
    run_jacobi_cuda_shared(h_A.data(), h_cuda_shared.data(), N, iterations, time_cuda_shared, 32, 8);
    auto v_cuda_s = validate_results(h_seq.data(), h_cuda_shared.data(), N);
    std::cout << "[CUDA Shared 32x8] Time: " << std::setw(8) << time_cuda_shared << " ms | Speedup: " 
              << std::setw(6) << (time_seq / time_cuda_shared) << "x | Max Abs Err: " << v_cuda_s.max_abs_error 
              << " | Max Rel Err: " << v_cuda_s.max_rel_error << "\n";

    std::cout << "========================================================================================\n";

    return 0;
}