# Introduction to Parallel Computing - Project Repository

**Student:** Riccardo Leone
**ID:** 246116
**Course:** Introduction to Parallel Computing (2025-2026) - University of Trento

This repository contains the source code and scripts for the course deliverables, focusing on **Sparse Matrix-Vector Multiplication (SpMV)** optimization.

## Project Structure

### 📂 [Deliverable 1: Sequential Implementation](./Deliverable_1_Sequential)
* **Focus:** Baseline sequential implementation of SpMV using CSR format.
* **Language:** C++
* **Status:** Completed.

### 📂 [Deliverable 2: Distributed Hybrid Implementation](./Deliverable_2_MPI)
* **Focus:** Distributed memory implementation using **MPI** combined with **OpenMP** for node-level parallelism (Hybrid Strategy).
* **Key Features:**
    * 1D Row-Cyclic Partitioning.
    * Hybrid Parallelism (MPI + OpenMP).
    * Advanced Metrics (Communication vs Computation breakdown).
    * Scalability analysis (Strong & Weak scaling).
* **Language:** C / MPI / OpenMP
* **Status:** **Final Submission.**

---
*See the `README.md` inside each folder for compilation and execution instructions.*