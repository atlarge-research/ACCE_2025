/*
* Description: Vector addition using OpenMP, adapted from the CUDA multi‐block example next week.
*
* To compile:
*     gcc -fopenmp -o vectorAdd_openmp vectorAdd_openmp.c
*
* Usage:
*     ./vectorAdd_openmp
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1 << 25)  // Total number of elements (e.g., 1M)

int main(void) {
    int i;
    double *A, *B, *C;
    double start_time, end_time, time_elapsed;
    
    // Allocate memory for vectors
    A = (double *) malloc(N * sizeof(double));
    B = (double *) malloc(N * sizeof(double));
    C = (double *) malloc(N * sizeof(double));
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error allocating memory.\n");
        return EXIT_FAILURE;
    }
    
    // Initialize input vectors
    for (i = 0; i < N; i++) {
        A[i] = (double) i;
        B[i] = (double)(N - i);
    }

    // Set the number of threads
    omp_set_num_threads(4);
    
    // Start the timer
    start_time = omp_get_wtime();
    
    // Parallel vector addition using OpenMP
    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    
    // End the timer
    end_time = omp_get_wtime();
    time_elapsed = end_time - start_time;
    
    // Print a few sample results for verification
    printf("Sample results:\n");
    for (i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }
    
    // Print the number of threads used
    printf("Number of threads: %d\n", omp_get_max_threads());
    // Calculate and print performance metrics
    printf("Time elapsed: %f seconds\n", time_elapsed);
    // Throughput: million elements processed per second
    printf("Throughput: %f million elements/second\n", (double)N / time_elapsed / 1e6);
    
    // Clean up
    free(A);
    free(B);
    free(C);
    
    return EXIT_SUCCESS;
}
 