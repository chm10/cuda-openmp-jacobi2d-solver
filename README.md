# Implementações de Equação de Laplace 2D pelo Método de Jacobi.

Este documento explica de forma detalhada as três abordagens implementadas no projeto, discutindo sua lógica, limitações e soluções aplicadas.

## Ambiente de Desenvolvimento (Local)

Você pode rodar as versões **Serial**, **OpenMP** e **CUDA** localmente e em HPC, desde que tenha Spack instalado para manter os pacotes compatíveis.

## Compatibilidade com projeto anterior

Carregue o ambiente do Spack conforme instruções abaixo para instalar as dependências (CMake, CUDA, GCC) de forma isolada.

```shell
# Baixe o spack se não tiver
git clone --depth=2 --branch=releases/v1.2 https://github.com/spack/spack.git ~/spack
# Carregue o ambiente com Spack
source ~/spack/share/spack/setup-env.sh
# Crie um ambiente novo
spack env create cenapad_challenge3
# Ative o novo ambiente
spack env activate cenapad_challenge3

# Adicionar dependências
spack add cmake
spack add cuda
spack add gcc

# Instalar e compilar no ambiente Spack
spack concretize -f
spack install

spack spec -l > spack_spec_concretized3.txt
```

## Estrutura do Projeto

Para manter o código organizado e facilitar a manutenção, o projeto segue a seguinte estrutura de diretórios:

```text
.
├── CMakeLists.txt          # Configuração de compilação (CMake + CUDA + OpenMP)
├── include/                # Arquivos de cabeçalho (.h / .cuh)
│   ├── common.h            # Funções auxiliares (init_matrix)
│   ├── jacobi_seq.h        # Protótipo da versão sequencial
│   ├── jacobi_omp.h        # Protótipos das versões OpenMP
│   ├── jacobi_cuda.cuh     # Protótipos das versões CUDA
│   └── validation.h        # Estrutura e protótipo de validação
├── src/                    # Implementações
│   ├── main.cpp            # Ponto de entrada e orquestração do benchmark
│   ├── jacobi_seq.cpp      # Versão sequencial (baseline)
│   ├── jacobi_omp.cpp      # Versões OpenMP (basic + tiled)
│   ├── jacobi_cuda.cu      # Versões CUDA (naive + shared memory)
│   └── validation.cpp      # Validação de resultados (erro absoluto/relativo)
├── doc/                    # Documentação adicional (relatório, figuras)
│   └── main.pdf            # Relatório completo do projeto
├── scripts/                # Scripts de automação
│   ├── run_experiments.sh  # Script para varredura de parâmetros (N, threads)
│   └── plot_results.py     # Geração de gráficos a partir dos resultados
├── build/                  # Diretório de compilação (gerado pelo CMake)
│   └── bin/
│       └── jacobi_benchmark  # Executável final
├── README.md               # Este arquivo
└── log_raw.txt             # Resultados dos experimentos (gerado por run_experiments.sh)
```

## Dependências

### Linux (nativo)
- **CMake** ≥ 3.18
- **CUDA Toolkit** ≥ 11.x (com suporte a compute capability 7.0+)
- **GCC** ou **Clang** com suporte a C++17
- **OpenMP** (incluído no GCC/Clang)

### Windows (via WSL2)
Para rodar este projeto no Windows, é necessário configurar o WSL2:

1. **Habilitar Modo de Desenvolvedor para gerar relatório Nvidia Nsight**:
   - Abra *Configurações Painel de Controle Nvidia* → *Para desenvolvedores* → *Habilite os contadores*

2. **Instalar WSL2**:
   ```powershell
   # No PowerShell (como Administrador)
   wsl --install
   wsl --set-default-version 2
   ```

3. **Instalar CUDA no WSL2**:
   Tente usar o Spack para evitar conflito de módulos e pacotes.

4. **Verificar acesso à GPU**:
   ```bash
   nvidia-smi   # Deve mostrar a GPU NVIDIA listada
   nvcc --version
   ```

### Ferramentas de Profiling
- **Nsight Systems** (`nsys`) — para profiling de sistema (via CUDA Toolkit ou instalador `.msi` no Windows)
- **Nsight Compute** (`ncu`) — para profiling de kernels (via CUDA Toolkit ou instalador `.msi` no Windows)

## Build

### Com CMake (Ambiente Spack)

```bash
# Criar diretório de build
rm -rf build && mkdir build && cd build

# Apontar compiladores do Spack e do sistema
SPACK_VIEW="$SPACK_ENV/.spack-env/view"
GXX_BIN=$(readlink -f "$(which g++)")

# Configurar com CMake
cmake .. \
  -DCMAKE_CXX_COMPILER="$GXX_BIN" \
  -DCMAKE_CUDA_COMPILER="$SPACK_VIEW/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$GXX_BIN" \
  -DCMAKE_CUDA_ARCHITECTURES=native

# Compilar
make -j$(nproc)
```

### Com CMake (Linux padrão, sem Spack)

```bash
# Criar diretório de build
mkdir -p build && cd build

# Configurar com CMake
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native

# Compilar
make -j$(nproc)
```

O executável será gerado em `build/bin/jacobi_benchmark`.

### Compilação manual com nvcc (para ajuste fino de flags)

A compilação manual permite ajustar a arquitetura da GPU e flags de otimização.
**Substitua `-arch=sm_XX` pela compute capability da sua placa:**

| GPU | Architecture Flag |
|-----|------------------|
| RTX 30xx (Ampere) | `-arch=sm_86` |
| RTX 40xx (Ada) | `-arch=sm_89` |
| RTX A-series (Ampere) | `-arch=sm_80` |
| GTX 10xx (Pascal) | `-arch=sm_61` |
| V100 (Volta) | `-arch=sm_70` |
| A100 (Ampere) | `-arch=sm_80` |

Volte para a raiz do projeto.
```bash
# Exemplo para GPU RTX 3090 com compute capability 8.6 (Ampere)
nvcc -O3 -arch=sm_86 -Isrc -Iinclude \
    src/main.cpp src/jacobi_cuda.cu src/jacobi_omp.cpp src/jacobi_seq.cpp src/validation.cpp \
    -o jacobi_benchmark \
    -Xcompiler "-fopenmp -O2"

# Executar
./jacobi_benchmark 4096 1000
```

> **Dica**: Use `nvidia-smi` para verificar a GPU e [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) para encontrar o compute capability correto.

## Execução

### Executar com parâmetros padrão (N=4096, 1000 iterações)

```bash
./build/bin/jacobi_benchmark
```

### Executar com parâmetros customizados

```bash
# N=4096, 500 iterações (configuração do relatório)
OMP_NUM_THREADS=8 ./build/bin/jacobi_benchmark 4096 500

# N=2048, 500 iterações
./build/bin/jacobi_benchmark 2048 500

# N=8192, 500 iterações
./build/bin/jacobi_benchmark 8192 500
```

### Executar varredura de experimentos

```bash
# Roda múltiplos valores de N e threads automaticamente
chmod +x scripts/run_experiments.sh
./scripts/run_experiments.sh
```

O script gera `log_raw.txt` e `results.csv` com todos os resultados.

## Saída do Benchmark

A saída padrão mostra uma tabela comparativa:

```text
========================================================================================
 Jacobi 2D Benchmark (N=4096, iterations=500)
========================================================================================
[CPU Seq]          Time:  1876.18 ms
[OpenMP Basic]     Time:  1501.96 ms | Speedup: 1.24915x | Max Abs Err: 0 | Max Rel Err: 0
[OpenMP Tiled]     Time:  1852.27 ms | Speedup: 1.01291x | Max Abs Err: 0 | Max Rel Err: 0
[CUDA Naive  8x 8] Time:   108.45 ms | Speedup:   17.3x | Max Abs Err: 2.40741e-35 | Max Rel Err: 2.40741e-26
[CUDA Naive 16x16] Time:  92.6081 ms | Speedup: 20.2593x | Max Abs Err: 2.40741e-35 | Max Rel Err: 2.40741e-26
[CUDA Naive 32x 8] Time:  92.1541 ms | Speedup: 20.3591x | Max Abs Err: 2.40741e-35 | Max Rel Err: 2.40741e-26
[CUDA Naive 32x16] Time:  93.7396 ms | Speedup: 20.0148x | Max Abs Err: 2.40741e-35 | Max Rel Err: 2.40741e-26
[CUDA Naive 32x32] Time:   111.62 ms | Speedup: 16.8086x | Max Abs Err: 2.40741e-35 | Max Rel Err: 2.40741e-26
[CUDA Shared 32x8] Time:  105.524 ms | Speedup: 17.7797x | Max Abs Err: 2.40741e-35 | Max Rel Err: 2.40741e-26
========================================================================================
```

### Colunas da saída

| Coluna | Descrição |
|--------|-----------|
| **Time** | Tempo de execução em milissegundos |
| **Speedup** | Speedup relativo à versão sequencial |
| **Max Abs Err** | Erro absoluto máximo vs. referência (sequencial) |
| **Max Rel Err** | Erro relativo máximo vs. referência (sequencial) |

## Implementações Detalhadas

### 1. CPU Sequencial (Baseline)
- Loop simples com ping-pong buffering (duas matrizes alternadas)
- Complexidade: $\mathcal{O}(N^2 \times \text{iterations})$
- Serve como referência para calcular speedups e corretude numérica

### 2. OpenMP Basic
- `#pragma omp for schedule(static)` distribui linhas entre threads
- `#pragma omp simd` instrui vetorização do loop interno
- Troca de ponteiros em região `#pragma omp single`

### 3. OpenMP Tiled
- Malha particionada em blocos (tiles) para melhorar localidade espacial nas caches L1/L2
- Cada thread processa um bloco completo antes da barreira da iteração
- Tile size padrão: 32×32

### 4. CUDA Naive
- Kernel direto com memória global
- Testa 5 geometrias de bloco: 8×8, 16×16, 32×8, 32×16, 32×32
- Ping-pong buffering no dispositivo

### 5. CUDA Shared Memory
- Tiles compartilhados de 34×10 com padding de 1 coluna (`[10][35]` ao invés de `[10][34]`)
- Carregamento colaborativo contíguo do bloco e halo (conforme analisado no relatório, esse mapeamento linear ainda gera *bank conflicts* na escrita e overhead de `__syncthreads()`)
- Configuração fixa: bloco 32×8, tile 34×10

## Validação

Todos os resultados são validados ponto a ponto contra a versão sequencial:
- **Erro absoluto máximo**: `max(|ref[i] - test[i]|)`
- **Erro relativo máximo**: `max(|ref[i] - test[i]| / (|ref[i]| + ε))` com `ε = 1e-9`

Na CPU (OpenMP), os resultados são bit a bit idênticos à referência (`Max Abs Err = 0`). Na GPU (CUDA), observa-se um resíduo numérico desprezível na ordem de `2.41e-35` em FP32 devido ao arredondamento das instruções *Fused Multiply-Add* (FMA) do hardware.

## Scripts

### `run_experiments.sh`
Varre múltiplos valores de N e threads:
```bash
# Variáveis configuráveis no script:
SIZES=(512 1024 2048 4096 8192)    # Dimensões da matriz
THREADS=(1 2 4 8 12 16)             # Número de threads OpenMP
ITERATIONS=500                      # Iterações por experimento
```

### `plot_results.py`
Gera gráficos a partir de `log_raw.txt` (requer Python com `matplotlib`, `seaborn`, `numpy` e `pandas`).

## Profiling e Relatórios

### Nsight Systems (nsys) — Profiling de Sistema

Gera relatórios de performance do sistema completo, incluindo tempo de execução CPU/GPU, transferências host-device e chamadas de CUDA API.

```bash
# Perfil básico (CUDA + NVTX)
nsys profile --trace=cuda,nvtx --force-overwrite true --stats=true \
    -o profile_system_jacobi ./build/bin/jacobi_benchmark 4096 500

# Com sudo (se necessário para acesso aos contadores da GPU)
sudo env "PATH=$PATH" nsys profile --trace=cuda,nvtx --force-overwrite true --stats=true \
    -o profile_system_jacobi ./build/bin/jacobi_benchmark 4096 500

# Com trace adicional de sistema operacional (osrt)
sudo nsys profile --trace=cuda,nvtx,osrt --stats=true \
    -o profile_system_jacobi ./build/bin/jacobi_benchmark 4096 500
```

**Análise do relatório:**
```bash
# Exibir estatísticas no terminal (ou abrir .nsys-rep pela interface gráfica)
nsys stats profile_system_jacobi.nsys-rep
```

### Nsight Compute (ncu) — Profiling de Kernels

Gera relatórios detalhados por kernel CUDA: occupancy, memory bandwidth, bank conflicts, roofine, etc.

```bash
# Perfil completo do kernel com fonte importada
sudo ncu --set full --import-source yes -o profile_kernels_jacobi \
    ./build/bin/jacobi_benchmark 4096 10

# Importar e visualizar resumo do relatório no terminal
ncu --import profile_kernels_jacobi.ncu-rep
```

### Comparação de Flags de Compilação

As flags de compilação impactam significativamente a performance. Para comparar diferentes arquiteturas CUDA:

```bash
# Testar diferentes architectures
nvcc -O3 -arch=sm_80 -Iinclude src/main.cpp src/jacobi_cuda.cu src/jacobi_omp.cpp src/jacobi_seq.cpp src/validation.cpp -o jacobi_benchmark_sm80 -Xcompiler "-fopenmp -O2"
nvcc -O3 -arch=sm_86 -Iinclude src/main.cpp src/jacobi_cuda.cu src/jacobi_omp.cpp src/jacobi_seq.cpp src/validation.cpp -o jacobi_benchmark_sm86 -Xcompiler "-fopenmp -O2"

# Comparar resultados
./jacobi_benchmark_sm80 4096 500
./jacobi_benchmark_sm86 4096 500
```

## Notas de Performance

- Use `nvidia-smi` para monitorar uso de GPU durante execução
- Use `ncu` (NVIDIA Nsight Compute) para profiling detalhado do kernel CUDA
- Use `nsys` (NVIDIA Nsight Systems) para profiling do sistema completo
- A versão com melhor speedup depende do hardware (número de núcleos físicos da CPU, largura de banda de memória e arquitetura da GPU)
- As flags `-O3` (CUDA) e `-O2` (host C++) mantêm a otimização de desempenho preservando a precisão numérica IEEE-754 FP32