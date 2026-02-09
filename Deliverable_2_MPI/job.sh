#!/bin/bash
#PBS -N SpMV_Hybrid_Demo
#PBS -q shortCPUQ
#PBS -l select=1:ncpus=4:mpiprocs=2:ompthreads=2
#PBS -l walltime=00:10:00

cd $PBS_O_WORKDIR
module purge
module load OpenMPI/4.1.4-GCC-12.2.0

echo "Compiling..."
mpicc -std=c99 -fopenmp spmv.c -o spmv

if [ ! -f "spmv" ]; then
    echo "Compilation failed!"
    exit 1
fi

export OMP_NUM_THREADS=2

echo "Running Hybrid SpMV (2 MPI Ranks x 2 OpenMP Threads)..."
mpirun -np 2 ./spmv 494_bus/494_bus.mtx

echo "Done."