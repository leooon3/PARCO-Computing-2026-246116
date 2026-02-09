#!/bin/bash
#PBS -N SpMV_Final_Benchmark
#PBS -o results_final.log
#PBS -e errors_final.log
#PBS -q short_cpuQ
#PBS -l select=1:ncpus=4:ompthreads=4:mem=1gb
#PBS -l walltime=0:20:00

# Load environment modules
module load gcc91
module load valgrind-3.15.0

cd "$PBS_O_WORKDIR"

# Compile the C++ code
g++ -O3 -fopenmp -std=c++11 main.cpp -o spmv_bench

# List of matrices to download and test
MATRICES=(
"https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz 1138_bus 1138_bus.mtx"
"https://suitesparse-collection-website.herokuapp.com/MM/HB/can_24.tar.gz can_24 can_24.mtx"
"https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk01.tar.gz bcsstk01 bcsstk01.mtx"
"https://suitesparse-collection-website.herokuapp.com/MM/HB/fs_183_1.tar.gz fs_183_1 fs_183_1.mtx"
"https://suitesparse-collection-website.herokuapp.com/MM/HB/ash85.tar.gz ash85 ash85.mtx"
)

echo "=== BENCHMARK START ==="

for entry in "${MATRICES[@]}"; do
    # Parse entry variables
    set -- $entry
    URL=$1; DIR=$2; FILE=$3
    MTX_PATH="$DIR/$FILE"

    echo ""
    echo ">>> PROCESSING MATRIX: $FILE <<<"

    # Download and extract if missing
    if [ ! -f "$MTX_PATH" ]; then
        echo "Downloading..."
        wget -q $URL
        tar -xzf "$DIR.tar.gz"
    fi

    # 1. Cache Profiling (Valgrind)
    echo "[Cache Analysis]"
    
    # CSR (Mode 1)
    LOG_CSR=$(valgrind --tool=cachegrind ./spmv_bench $MTX_PATH 0.0 1 1 2>&1)
    MISS_CSR=$(echo "$LOG_CSR" | grep "D1  misses" | awk '{print $4}')
    echo "CSR D1 Misses: $MISS_CSR"

    # COO (Mode 2)
    LOG_COO=$(valgrind --tool=cachegrind ./spmv_bench $MTX_PATH 0.0 1 2 2>&1)
    MISS_COO=$(echo "$LOG_COO" | grep "D1  misses" | awk '{print $4}')
    echo "COO D1 Misses: $MISS_COO"

    # 2. Performance (10 Runs for 90th Percentile)
    echo "[Performance 10 Runs (ms)]"
    echo "Run | 1 Thread | 4 Threads"
    
    for i in {1..10}; do
        # 1 Thread (Mode 0)
        OUT_1=$(./spmv_bench $MTX_PATH 0.0 1 0)
        # Fix: Grep specifically for "Time:" to get the number
        T1=$(echo "$OUT_1" | grep "Time:" | awk '{print $2}')

        # 4 Threads (Mode 0)
        OUT_4=$(./spmv_bench $MTX_PATH 0.0 4 0)
        # Fix: Grep specifically for "Time:"
        T4=$(echo "$OUT_4" | grep "Time:" | awk '{print $2}')

        echo "$i | $T1 | $T4"
    done
done

echo ""
echo "=== BENCHMARK COMPLETE ==="
