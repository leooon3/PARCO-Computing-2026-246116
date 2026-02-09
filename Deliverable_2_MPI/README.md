# Deliverable 2: Distributed SpMV with MPI + OpenMP

## Description
This project implements a distributed Sparse Matrix-Vector Multiplication solver using MPI for inter-node communication and OpenMP for intra-node parallelism (Hybrid Strategy). It uses 1D Row-Cyclic partitioning.

## Files
- `spmv.c`: Main source code (Hybrid MPI+OpenMP).
- `job.sh`: PBS script for execution on the University HPC3 cluster.

## Compilation
To compile the code with Hybrid support:
```bash
module load OpenMPI/4.1.4-GCC-12.2.0
mpicc -std=c99 -fopenmp spmv.c -o spmv
Execution
Run on the cluster using the provided script or manually:

Bash

# Example: 2 MPI Processes, 2 OpenMP Threads each
export OMP_NUM_THREADS=2
mpirun -np 2 ./spmv <matrix_file.mtx>