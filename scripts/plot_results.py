import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração visual global
sns.set_theme(style="whitegrid", palette="tab10")

# -----------------------------------------------------------------------------
# 1. Função para Ler e Processar o Log do Benchmark (log_raw.txt)
# -----------------------------------------------------------------------------
def load_benchmark_data(log_filepath='log_raw.txt'):
    with open(log_filepath, 'r') as f:
        text = f.read()

    blocks = text.split('========================================================================================')
    records = []
    current_N = None

    for block in blocks:
        n_match = re.search(r'Jacobi 2D Benchmark \(N=(\d+)', block)
        if n_match:
            current_N = int(n_match.group(1))
            continue
        
        lines = block.strip().split('\n')
        for line in lines:
            if not line.strip(): 
                continue
            m = re.match(r'\[(.*?)\]\s+Time:\s+([\d\.]+)\s*ms(?: \| Speedup:\s+([\d\.]+)x)?', line)
            if m and current_N:
                impl = m.group(1).strip()
                time_ms = float(m.group(2))
                records.append({'N': current_N, 'Implementation': impl, 'Time_ms': time_ms})

    df = pd.DataFrame(records)
    
    # Calcular média das execuções por tamanho de N e Implementação
    df_avg = df.groupby(['N', 'Implementation'])['Time_ms'].mean().reset_index()

    # Calcular Speedup relativo à média da versão Sequencial
    seq_times = df_avg[df_avg['Implementation'] == 'CPU Seq'].set_index('N')['Time_ms'].to_dict()
    df_avg['Speedup'] = df_avg.apply(lambda r: seq_times[r['N']] / r['Time_ms'], axis=1)

    return df_avg, seq_times


# -----------------------------------------------------------------------------
# 2. Função para Extrair Speedup OpenMP por Thread (usa marcadores de log_raw.txt)
# -----------------------------------------------------------------------------
def load_omp_speedup_by_thread(log_filepath='log_raw.txt'):
    """Extrai speedup do OpenMP Basic e Tiled agrupado por número de threads."""
    with open(log_filepath, 'r') as f:
        text = f.read()

    # Encontrar todos os blocos marcados com "# Threads: N | ..."
    thread_blocks = re.split(r'(?=# Threads:)', text)
    
    omp_data = {'threads': [], 'OpenMP Basic': [], 'OpenMP Tiled': []}
    
    for block in thread_blocks:
        threads_match = re.search(r'# Threads:\s*(\d+)', block)
        if not threads_match:
            continue
        threads = int(threads_match.group(1))
        
        basic_match = re.search(r'\[OpenMP Basic\]\s+Time:\s*[\d.]+\s+ms\s*\|\s*Speedup:\s*([\d.]+)', block)
        tiled_match = re.search(r'\[OpenMP Tiled\]\s+Time:\s*[\d.]+\s+ms\s*\|\s*Speedup:\s*([\d.]+)', block)
        
        if basic_match:
            omp_data['threads'].append(threads)
            omp_data['OpenMP Basic'].append(float(basic_match.group(1)))
        if tiled_match:
            omp_data['OpenMP Tiled'].append(float(tiled_match.group(1)))

    # Remover duplicatas mantendo ordem (pode haver múltiplos N com mesma thread count)
    seen = set()
    unique_threads = []
    for t in omp_data['threads']:
        if t not in seen:
            seen.add(t)
            unique_threads.append(t)

    return unique_threads, omp_data


# Carregar dados tratados
df_avg, seq_times, raw_records = load_benchmark_data('log_raw.txt')
Ns = sorted(df_avg['N'].unique())

# Carregar speedup OpenMP por thread (se disponível no log)
omp_threads, omp_speedup_data = load_omp_speedup_by_thread('log_raw.txt')

# -----------------------------------------------------------------------------
# GRÁFICO 1: Tempo de Execução vs. Tamanho do Problema N (Escala Logarítmica)
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
key_impls = ['CPU Seq', 'OpenMP Basic', 'OpenMP Tiled', 'CUDA Naive 32x 8', 'CUDA Shared 32x8']
markers = ['o', 's', '^', 'D', 'v']

for impl, marker in zip(key_impls, markers):
    sub = df_avg[df_avg['Implementation'] == impl]
    plt.plot(sub['N'], sub['Time_ms'], marker=marker, linewidth=2, label=impl)

plt.xscale('log', base=2)
plt.yscale('log')
plt.xticks(Ns, labels=[str(n) for n in Ns])
plt.title('Tempo de Execução vs. Tamanho do Problema (N) [Escala Log-Log]', fontsize=12, fontweight='bold')
plt.xlabel('Tamanho da Matriz N (N x N)', fontsize=10)
plt.ylabel('Tempo de Execução (ms)', fontsize=10)
plt.legend(title='Implementação', frameon=True)
plt.tight_layout()
plt.savefig('1_tempo_vs_N_log.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# GRÁFICO 2: Evolução do Speedup em Função de N
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
eval_impls = ['OpenMP Basic', 'OpenMP Tiled', 'CUDA Naive 32x 8', 'CUDA Shared 32x8']
eval_markers = ['s', '^', 'D', 'v']

for impl, marker in zip(eval_impls, eval_markers):
    sub = df_avg[df_avg['Implementation'] == impl]
    plt.plot(sub['N'], sub['Speedup'], marker=marker, linewidth=2, label=impl)

plt.axhline(1.0, color='gray', linestyle='--', label='Baseline CPU Seq (1x)')
plt.xticks(Ns, labels=[str(n) for n in Ns])
plt.title('Evolução do Speedup com o Aumento do Tamanho da Matriz (N)', fontsize=12, fontweight='bold')
plt.xlabel('Tamanho da Matriz N (N x N)', fontsize=10)
plt.ylabel('Speedup (T_seq / T_par)', fontsize=10)
plt.legend(title='Implementação', frameon=True)
plt.tight_layout()
plt.savefig('2_speedup_vs_N.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# GRÁFICO 3: Impacto das Geometrias de Bloco CUDA (N = 8192)
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
cuda_sub = df_avg[(df_avg['N'] == 8192) & (df_avg['Implementation'].str.contains('CUDA'))].copy()
cuda_sub['Label'] = cuda_sub['Implementation'].str.replace('CUDA Naive ', '').str.replace('CUDA Shared ', 'Shared ')

bars = plt.bar(cuda_sub['Label'], cuda_sub['Time_ms'], color=sns.color_palette("Blues_r", len(cuda_sub)))
plt.title('Comparativo de Geometrias de Bloco e Kernel CUDA (N = 8192)', fontsize=12, fontweight='bold')
plt.xlabel('Configuração de Bloco / Kernel', fontsize=10)
plt.ylabel('Tempo de Execução (ms)', fontsize=10)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.1f} ms', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('3_cuda_block_geometries.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# GRÁFICO 4: Comparação Direta de Desempenho para N = 4096
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5.5))
n4096_sub = df_avg[df_avg['N'] == 4096].sort_values(by='Time_ms', ascending=False)

bars = plt.barh(n4096_sub['Implementation'], n4096_sub['Time_ms'], color=sns.color_palette("viridis", len(n4096_sub)))
plt.title('Comparação Direta de Tempo de Execução para N = 4096', fontsize=12, fontweight='bold')
plt.xlabel('Tempo de Execução em ms (Menor é melhor)', fontsize=10)
plt.ylabel('Versão / Otimização', fontsize=10)

for bar in bars:
    xval = bar.get_width()
    plt.text(xval + 20, bar.get_y() + bar.get_height()/2.0, f'{xval:.1f} ms ({seq_times[4096]/xval:.1f}x)', ha='left', va='center', fontsize=9)

plt.xlim(0, max(n4096_sub['Time_ms']) * 1.22)
plt.tight_layout()
plt.savefig('4_comparativo_algoritmos_N4096.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# GRÁFICO 5: Escalabilidade OpenMP vs. Número de Threads (N = 4096)
# -----------------------------------------------------------------------------
if omp_threads and 'OpenMP Basic' in omp_speedup_data:
    threads = omp_speedup_data['threads']
    speedup_omp_basic = omp_speedup_data['OpenMP Basic']
    speedup_omp_tiled = omp_speedup_data.get('OpenMP Tiled', [])

    plt.figure(figsize=(8, 5))
    plt.plot(threads, speedup_omp_basic, marker='o', label='OpenMP Basic', linewidth=2)
    if speedup_omp_tiled:
        plt.plot(threads, speedup_omp_tiled, marker='s', label='OpenMP Tiled', linewidth=2)
    plt.plot(threads, threads, '--', color='gray', label='Speedup Ideal (Linear)')

    plt.title('Escalabilidade OpenMP vs. Número de Threads (N=4096)', fontsize=12, fontweight='bold')
    plt.xlabel('Número de Threads', fontsize=10)
    plt.ylabel('Speedup (T_seq / T_par)', fontsize=10)
    plt.xticks(threads)
    plt.legend()
    plt.tight_layout()
    plt.savefig('5_scalability_openmp.png', dpi=300)
    plt.close()
else:
    print("Aviso: dados de threads não encontrados em log_raw.txt. Usando valores padrão.")

print("Todos os 5 gráficos foram gerados e salvos com sucesso!")