cmake_minimum_required(VERSION 3.18)
project(Jacobi2DOptimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Ativação do OpenMP para CPU
find_package(OpenMP REQUIRED)

# Diretórios de cabeçalho
include_directories(include)

# Código-fonte
set(SOURCES
    src/main.cpp
    src/jacobi_seq.cpp
    src/jacobi_omp.cpp
    src/jacobi_cuda.cu
    src/validation.cpp
)

add_executable(jacobi_benchmark ${SOURCES})

# Flags de otimização de compilador
target_compile_options(jacobi_benchmark PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

target_link_libraries(jacobi_benchmark PRIVATE OpenMP::OpenMP_CXX)

# Diretório de saída dos binários
set_target_properties(jacobi_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)