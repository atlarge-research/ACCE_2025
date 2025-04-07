#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel function for vector addition
__global__ void vectorAdd(const int *A, const int *B, int *C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Compute global index
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    int numElements = 1 << 20; // Increase the number of elements to 2^20
    size_t size = numElements * sizeof(int);

    // Allocate memory on the GPU using Unified Memory
    int *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < numElements; ++i) {
        A[i] = i;
        B[i] = i;
    }

    // Adjust the number of threads per block and blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);

    cudaDeviceSynchronize();

    for (int i = 0; i < numElements; ++i) {
        if (C[i] != A[i] + B[i]) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    printf("Test PASSED\n");

    return 0;
}
