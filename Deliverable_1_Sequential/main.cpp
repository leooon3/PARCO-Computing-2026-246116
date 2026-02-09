#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <sstream>

// --- DATA STRUCTURES ---

struct MatrixCOO {
    int rows, cols, nnz;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;
};

struct MatrixCSR {
    int rows, cols, nnz;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

// --- UTILS ---

// Read Matrix Market file (.mtx)
MatrixCOO read_mtx(const std::string& filename) {
    std::ifstream file(filename);
    MatrixCOO mat;
    if (!file.is_open()) exit(1);

    std::string line;
    // Skip comments
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        break; 
    }

    std::stringstream ss(line);
    ss >> mat.rows >> mat.cols >> mat.nnz;

    mat.row_indices.reserve(mat.nnz);
    mat.col_indices.reserve(mat.nnz);
    mat.values.reserve(mat.nnz);

    int r, c;
    double val;
    while (file >> r >> c) {
        mat.row_indices.push_back(r - 1); // 1-based to 0-based
        mat.col_indices.push_back(c - 1);
        if (file >> val) mat.values.push_back(val);
        else { file.clear(); mat.values.push_back(1.0); }
    }
    return mat;
}

// Generate random COO matrix
MatrixCOO generate_random_coo(int rows, int cols, double density) {
    MatrixCOO mat = {rows, cols, 0};
    int est_nnz = rows * cols * density;
    
    mat.row_indices.reserve(est_nnz);
    mat.col_indices.reserve(est_nnz);
    mat.values.reserve(est_nnz);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (dis(gen) < density) {
                mat.row_indices.push_back(i);
                mat.col_indices.push_back(j);
                mat.values.push_back(1.0);
            }
        }
    }
    mat.nnz = mat.values.size();
    return mat;
}

// Convert COO to CSR
MatrixCSR coo_to_csr(const MatrixCOO& coo) {
    MatrixCSR csr;
    csr.rows = coo.rows; csr.cols = coo.cols; csr.nnz = coo.nnz;
    csr.values = coo.values;
    csr.col_indices = coo.col_indices;
    csr.row_ptr.resize(csr.rows + 1, 0);

    for (int r : coo.row_indices) csr.row_ptr[r + 1]++;
    for (int i = 0; i < csr.rows; ++i) csr.row_ptr[i + 1] += csr.row_ptr[i];
    
    return csr;
}

// --- KERNELS ---

// Serial SpMV (COO)
void spmv_coo_serial(const MatrixCOO& mat, const std::vector<double>& x, std::vector<double>& y) {
    std::fill(y.begin(), y.end(), 0.0);
    for (int i = 0; i < mat.nnz; ++i) {
        y[mat.row_indices[i]] += mat.values[i] * x[mat.col_indices[i]];
    }
}

// Parallel SpMV (CSR)
void spmv_csr_parallel(const MatrixCSR& mat, const std::vector<double>& x, std::vector<double>& y) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < mat.rows; ++i) {
        double sum = 0.0;
        for (int j = mat.row_ptr[i]; j < mat.row_ptr[i+1]; ++j) {
            sum += mat.values[j] * x[mat.col_indices[j]];
        }
        y[i] = sum;
    }
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    if (argc < 4) return 1;

    std::string input = argv[1];
    int threads = std::atoi(argv[3]);
    int mode = std::atoi(argv[4]); // 0=Run, 1=CSR_Valgrind, 2=COO_Valgrind
    
    omp_set_num_threads(threads);
    MatrixCOO coo;

    // Load or Generate Data
    if (input.find(".mtx") != std::string::npos) {
        std::cout << "Reading: " << input << std::endl;
        coo = read_mtx(input);
    } else {
        int N = std::atoi(argv[1]);
        double dens = std::atof(argv[2]);
        std::cout << "Generating: " << N << "x" << N << " d=" << dens << std::endl;
        coo = generate_random_coo(N, N, dens);
    }

    MatrixCSR csr = coo_to_csr(coo);
    std::vector<double> x(coo.cols, 1.0), y(coo.rows, 0.0);

    // Warmup
    if (mode == 0) spmv_csr_parallel(csr, x, y);

    auto start = std::chrono::high_resolution_clock::now();

    if (mode == 2) spmv_coo_serial(coo, x, y);
    else spmv_csr_parallel(csr, x, y);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    return 0;
}
