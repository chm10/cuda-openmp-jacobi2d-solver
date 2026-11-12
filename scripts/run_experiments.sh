#!/bin/bash
set -e

OUTPUT_CSV="results.csv"
LOG_FILE="log_raw.txt"
echo "Experiment,N,Threads,Implementation,Time_ms,Speedup" > "$OUTPUT_CSV"
: > "$LOG_FILE"

SIZES=(512 1024 2048 4096 8192)
THREADS=(1 2 4 8 12 16)
ITERATIONS=500
REPEATS=5

echo "1) Executando escalabilidade por tamanho de matriz (OMP_NUM_THREADS=8, $REPEATS repeticoes)..."
export OMP_NUM_THREADS=8
for N in "${SIZES[@]}"; do
    for R in $(seq 1 $REPEATS); do
        echo "----------------------------------------------------------------------------------------" >> "$LOG_FILE"
        echo "# Experiment: size_scaling | Repeat: $R | Threads: 8 | N: $N" >> "$LOG_FILE"
        echo "----------------------------------------------------------------------------------------" >> "$LOG_FILE"

        ./build/bin/jacobi_benchmark "$N" "$ITERATIONS" 2>&1 | tee -a "$LOG_FILE" | \
            grep -E '^\[' | \
            sed -E 's/^\[([^]]+)\][[:space:]]+Time:[[:space:]]+([0-9.]+)[[:space:]]+ms(\|[[:space:]]+Speedup:[[:space:]]+([0-9.]+)x)?/size_scaling,'"$N"',8,\1,\2,\4/g' >> "$OUTPUT_CSV"
    done
done

echo "2) Executando escalabilidade OpenMP por numero de threads (N=4096)..."
for T in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$T
    echo "----------------------------------------------------------------------------------------" >> "$LOG_FILE"
    echo "# Experiment: thread_scaling | Threads: $T | N: 4096" >> "$LOG_FILE"
    echo "----------------------------------------------------------------------------------------" >> "$LOG_FILE"

    ./build/bin/jacobi_benchmark 4096 "$ITERATIONS" 2>&1 | tee -a "$LOG_FILE" | \
        grep -E '^\[' | \
        sed -E 's/^\[([^]]+)\][[:space:]]+Time:[[:space:]]+([0-9.]+)[[:space:]]+ms(\|[[:space:]]+Speedup:[[:space:]]+([0-9.]+)x)?/thread_scaling,4096,'"$T"',\1,\2,\4/g' >> "$OUTPUT_CSV"
done

echo "Experimentos concluidos! Logs em $LOG_FILE e tabela consolidada em $OUTPUT_CSV"