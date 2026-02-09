#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h> // BONUS HYBRID: Libreria OpenMP

/* --- STRUTTURE DATI --- */
typedef struct {
    int r, c;
    double val;
} COOElement;

typedef struct {
    int *row_ptr;
    int *col_ind;
    double *val;
    int local_rows; 
    int total_cols; 
    int nnz;       
} CSRMatrix;

/* --- FUNZIONI UTILI --- */
void allocate_csr(CSRMatrix *mat, int local_m, int nnz, int total_n) {
    mat->local_rows = local_m;
    mat->total_cols = total_n;
    mat->nnz = nnz;
    mat->row_ptr = (int*)calloc(local_m + 1, sizeof(int));
    mat->col_ind = (int*)malloc(nnz * sizeof(int));
    mat->val = (double*)malloc(nnz * sizeof(double));
}

void free_csr(CSRMatrix *mat) {
    if(mat->row_ptr) free(mat->row_ptr);
    if(mat->col_ind) free(mat->col_ind);
    if(mat->val) free(mat->val);
}

void coo_to_csr(COOElement *elements, int count, CSRMatrix *mat, int size) {
    for (int i = 0; i < count; i++) {
        int local_row = elements[i].r / size; 
        mat->row_ptr[local_row + 1]++;
    }
    for (int i = 0; i < mat->local_rows; i++) {
        mat->row_ptr[i + 1] += mat->row_ptr[i];
    }
    int *current_pos = (int*)calloc(mat->local_rows, sizeof(int));
    for (int i = 0; i < mat->local_rows; i++) current_pos[i] = mat->row_ptr[i];

    for (int i = 0; i < count; i++) {
        int local_row = elements[i].r / size;
        int dest = current_pos[local_row];
        mat->col_ind[dest] = elements[i].c;
        mat->val[dest] = elements[i].val;
        current_pos[local_row]++;
    }
    free(current_pos);
}

// --- KERNEL DI CALCOLO SpMV IBRIDO ---
void spmv_csr_kernel(CSRMatrix *mat, double *x, double *y_local) {
    // BONUS: Parallelismo Ibrido (OpenMP all'interno del processo MPI)
    // Distribuisce il loop esterno tra i thread disponibili
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < mat->local_rows; i++) {
        double sum = 0.0;
        int start = mat->row_ptr[i];
        int end = mat->row_ptr[i+1];
        
        for (int j = start; j < end; j++) {
            int col = mat->col_ind[j];
            sum += mat->val[j] * x[col];
        }
        y_local[i] = sum;
    }
}

/* --- MAIN --- */
int main(int argc, char *argv[]) {
    // Inizializza MPI con supporto Thread (MPI_THREAD_FUNNELED è il minimo per Hybrid)
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M, N, global_nnz;
    int *send_counts = NULL; 
    int *displs = NULL;
    COOElement *recv_buffer = NULL;
    COOElement *send_buffer = NULL;

    /* 1. LETTURA (Solo Rank 0) */
    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <matrix.mtx>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        FILE *f = fopen(argv[1], "r");
        if (!f) { perror("File error"); MPI_Abort(MPI_COMM_WORLD, 1); }
        
        char line[1024];
        do { if (!fgets(line, 1024, f)) break; } while (line[0] == '%');
        sscanf(line, "%d %d %d", &M, &N, &global_nnz);
        
        if(rank==0) printf("[Master] Hybrid MPI+OpenMP. Matrice %dx%d, NNZ=%d. Processi=%d\n", M, N, global_nnz, size);

        COOElement *all = (COOElement*)malloc(global_nnz * sizeof(COOElement));
        send_counts = (int*)calloc(size, sizeof(int));

        for (int i = 0; i < global_nnz; i++) {
            int r, c; double v;
            fscanf(f, "%d %d %lf", &r, &c, &v);
            r--; c--; 
            all[i].r = r; all[i].c = c; all[i].val = v;
            send_counts[r % size]++; 
        }
        fclose(f);

        send_buffer = (COOElement*)malloc(global_nnz * sizeof(COOElement));
        int *offsets = (int*)calloc(size, sizeof(int));
        displs = (int*)calloc(size, sizeof(int));
        
        displs[0] = 0;
        for (int i = 1; i < size; i++) displs[i] = displs[i-1] + send_counts[i-1];
        memcpy(offsets, displs, size * sizeof(int));

        for (int i = 0; i < global_nnz; i++) {
            int owner = all[i].r % size;
            send_buffer[offsets[owner]++] = all[i];
        }
        free(all); free(offsets);
    }

    /* 2. DISTRIBUZIONE DATI */
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int my_nnz;
    MPI_Scatter(send_counts, 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Datatype MPI_COO;
    MPI_Type_contiguous(sizeof(COOElement), MPI_BYTE, &MPI_COO);
    MPI_Type_commit(&MPI_COO);

    recv_buffer = (COOElement*)malloc(my_nnz * sizeof(COOElement));
    MPI_Scatterv(send_buffer, send_counts, displs, MPI_COO, 
                 recv_buffer, my_nnz, MPI_COO, 0, MPI_COMM_WORLD);
    
    int local_rows = M / size + ((rank < M % size) ? 1 : 0);
    CSRMatrix local_mat;
    allocate_csr(&local_mat, local_rows, my_nnz, N);
    coo_to_csr(recv_buffer, my_nnz, &local_mat, size);
    free(recv_buffer); 

    /* 3. PREPARAZIONE VETTORI */
    double *x = (double*)malloc(N * sizeof(double));
    double *y_local = (double*)malloc(local_rows * sizeof(double));

    if (rank == 0) {
        for(int i=0; i<N; i++) x[i] = 1.0; 
    }
    
    /* 4. ESECUZIONE E TIMER AVANZATO (BONUS METRICHE) */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_total = MPI_Wtime();
    
    // --- FASE 1: COMUNICAZIONE (Broadcast Vettore X) ---
    // Timer dedicato per la comunicazione
    double t_comm_start = MPI_Wtime();
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_comm_end = MPI_Wtime();
    
    // --- FASE 2: CALCOLO (SpMV Locale Ibrido) ---
    // Il calcolo avviene in parallelo coi thread OpenMP
    spmv_csr_kernel(&local_mat, x, y_local);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end_total = MPI_Wtime();

    double t_total = t_end_total - t_start_total;
    double t_comm = t_comm_end - t_comm_start;
    double t_comp = t_total - t_comm; // Tempo residuo è calcolo

    /* 5. RACCOLTA RISULTATI */
    double local_sum = 0.0;
    for(int i=0; i<local_rows; i++) local_sum += y_local[i];
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== RISULTATI DETTAGLIATI (Bonus Metrics) ===\n");
        printf("Tempo Totale:        %f s\n", t_total);
        printf("Tempo Comunicazione: %f s (%.2f%%)\n", t_comm, (t_comm/t_total)*100.0);
        printf("Tempo Calcolo:       %f s (%.2f%%)\n", t_comp, (t_comp/t_total)*100.0);
        printf("Check Somma Y:       %f\n", global_sum);
        printf("FLOPS stimati:       %e\n", (2.0 * global_nnz) / t_total);
    }

    free(x); free(y_local); free_csr(&local_mat);
    if(rank==0) { free(send_counts); free(displs); free(send_buffer); }
    MPI_Type_free(&MPI_COO);
    
    MPI_Finalize();
    return 0;
}