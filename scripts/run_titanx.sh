#!/bin/bash
#SBATCH --partition=titanx
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=jacobi_titanx
#SBATCH --output=jacobi_titanx_%j.log

export OMP_NUM_THREADS=8

./build/bin/jacobi_benchmark 4096 1000