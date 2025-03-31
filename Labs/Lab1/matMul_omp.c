#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static double time_secs() {
    struct timeval tv;

    if (gettimeofday(&tv, 0) != 0) {
        fprintf(stderr, "could not do timing\n");
        exit(1);
    }

    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

void init(float *A, float *B, float *C, float *D, int N, int K, int M) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            A[i * K + j] = 1;

    for (int i = 0; i < K; i++)
        for (int j = 0; j < M; j++)
            B[i * M + j] = 2;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) {
            C[i * M + j] = 0;
            D[i * M + j] = 0;
        }
}

void validate(float *A, float *B, int N, int M) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            if (fabs(A[i * M + j] - B[i * M + j]) > 1e-6) {
                printf("results does not match\n");
                exit(1);
            }
}

double matMul_seq(float *A, float *B, float *C, int N, int K, int M) {
    double start_time_seq, end_time_seq;

    start_time_seq = time_secs();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float res = 0;
            for (int k = 0; k < K; k++)
                res += A[i * K + k] * B[k * M + j];
            C[i * M + j] = res;
        }
    }

    end_time_seq = time_secs();

    return end_time_seq - start_time_seq;
}

double matMul_par(float *A, float *B, float *C, int N, int K, int M) {
    double start_time_seq, end_time_seq;

#pragma omp parallel
    {
        // Only have one thread print it (to avoid multiple prints)
        if (omp_get_thread_num() == 0)
            printf("Number of threads used: %d\n", omp_get_num_threads());
    }

    start_time_seq = time_secs();

// TODO: implement the matrix multiplication,
// `     parallelize the matrix multiplication using OpenMP
// Hint: use collapse(2) clause to parallelize the nested loops


    end_time_seq = time_secs();

    return end_time_seq - start_time_seq;
}

int main(int argc, char **argv) {
    int N = 1 << 10;
    int K = 1 << 7;
    int M = 1 << 10;

    double elapsed_time_seq, elapsed_time_par;

    float *A = (float *)malloc(N * K * sizeof(float));
    float *B = (float *)malloc(K * M * sizeof(float));
    float *results_seq = (float *)malloc(N * M * sizeof(float));
    float *results_omp = (float *)malloc(N * M * sizeof(float));

    // initialize matrices
    init(A, B, results_seq, results_omp, N, K, M);

    elapsed_time_seq = matMul_seq(A, B, results_seq, N, K, M);

    elapsed_time_par = matMul_par(A, B, results_omp, N, K, M);

    validate(results_seq, results_omp, N, M);

    printf("TEST PASSED\n");

    printf("seq time:%f\npar time:%f\nspeed up:%f\n", elapsed_time_seq, elapsed_time_par, elapsed_time_seq / elapsed_time_par);

    free(A);
    free(B);
    free(results_seq);
    free(results_omp);

    return 0;
}
