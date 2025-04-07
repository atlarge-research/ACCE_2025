#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel function for vector addition
__global__ void vectorAdd(const int *A, const int *B, int *C, int numElements) {
    int i = threadIdx.x; // TODO: Compute global index
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    int numElements = 1 << 20; // Increase the number of elements to 2^20
    size_t size = numElements * sizeof(int);

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

    // TODO: Adjust the number of threads per block and blocks per grid
    // the grid dimension should > 1
    vectorAdd<<<1, numElements>>>(d_A, d_B, d_C, numElements);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError(); // Check if the current CUDA driver has returned any errors. Remember to call cudaDeviceSynchronize() before this statement.
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }

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