/*
* Description: Sequential vector addition.
*
* To compile:
*     gcc -o vectorAdd_sequential vectorAdd_sequential.c
*
* Usage:
*     ./vectorAdd_sequential
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1 << 25)  // Total number of elements (e.g., 1M)

int main(void) {
    int i;
    double *A, *B, *C;
    clock_t start_time, end_time;
    double time_elapsed;

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

    // Start the timer
    start_time = clock();

    // Sequential vector addition
    for (i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    // End the timer
    end_time = clock();
    time_elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Print a few sample results for verification
    printf("Sample results:\n");
    for (i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

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