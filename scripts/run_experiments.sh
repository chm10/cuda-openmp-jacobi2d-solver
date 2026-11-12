#!/bin/bash

OUTPUT_CSV="results.csv"
echo "N,Threads,Implementation,Time_ms,Speedup" > $OUTPUT_CSV

SIZES=(512 1024 2048 4096 8192)
THREADS=(1 2 4 8 12 16)
ITERATIONS=500

echo "Iniciando varredura de experimentos..."

for N in "${SIZES[@]}"; do
    for T in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$T
        echo "----------------------------------------------------------------------------------------" >> log_raw.txt
        echo "# Threads: $T | N: $N" >> log_raw.txt
        echo "----------------------------------------------------------------------------------------" >> log_raw.txt
        
        # Executar benchmark e salvar resultados no CSV
        ./build/bin/jacobi_benchmark $N $ITERATIONS 2>&1 | tee -a log_raw.txt | \
            grep -E '^\[|Threads:' | \
            sed -E 's/.*\[([^]]+)\].*Time:\s+([0-9.]+).*Speedup:\s+([0-9.]+)/'"$N,$T,\1,\2,\3"'/g' >> $OUTPUT_CSV
    done
done

echo "Experimentos concluídos! Logs salvos em log_raw.txt e resultados em $OUTPUT_CSV"