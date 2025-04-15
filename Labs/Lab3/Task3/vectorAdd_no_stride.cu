#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA Kernel function for vector addition without grid stride
__global__ void vectorAdd(const int *A, const int *B, int *C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Compute global index

    if (i < numElements) { // Ensure thread index is within bounds
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv) {
    // Check command-line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s blk_dim\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int numElements = 1 << 20; // Increase the number of elements to 2^20
    size_t size = numElements * sizeof(int);

    // Parse grid and block dimensions from command line
    int blk_dim = atoi(argv[1]);

    if (blk_dim <= 0) {
        fprintf(stderr, "blk_dim must be positive integers.\n");
        exit(EXIT_FAILURE);
    }

    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
        h_B[i] = i;
    }

    int *d_A = NULL;
    cudaMalloc((void **)&d_A, size);
    int *d_B = NULL;
    cudaMalloc((void **)&d_B, size);
    int *d_C = NULL;
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with user-specified grid and block dimensions
    int grid_dim = (numElements + blk_dim - 1) / blk_dim; // Calculate grid size

    vectorAdd<<<grid_dim, blk_dim>>>(d_A, d_B, d_C, numElements);

    // Check for kernel launch errors
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(kernelErr));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < numElements; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Test PASSED\n");

    return 0;
}
