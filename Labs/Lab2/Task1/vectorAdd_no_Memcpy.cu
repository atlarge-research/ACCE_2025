#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel function for vector addition
__global__ void vectorAdd(const int *A, const int *B, int *C, int numElements) {
    int i = threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    // Define the size of the vector
    int numElements = 500;
    size_t size = numElements * sizeof(int);
    
    // Ensure numElements does not exceed the maximum number of threads per block
    int maxThreadsPerBlock = 1024;
    if (numElements > maxThreadsPerBlock) {
        fprintf(stderr, "Error: numElements exceeds the maximum number of threads per block (%d)\n", maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    // Allocate memory on the host
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    // Initialize the vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
        h_B[i] = i;
    }

    // TODO: Allocate memory on the GPU
    

    // TODO: Copy the vectors from the host to the GPU
    

    // Launch the kernel function with one block and numElements threads
    vectorAdd<<<1, numElements>>>(d_A, d_B, d_C, numElements);

    // Synchronize the device to ensure all threads have completed
    cudaDeviceSynchronize();

    // TODO: Copy the result from the GPU back to the host
    

    // Check the result with a tolerance
    float tolerance = 1e-3;
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > tolerance) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            fprintf(stderr, "h_C[%d] = %d, h_A[%d] = %d, h_B[%d] = %d\n", i, h_C[i], i, h_A[i], i, h_B[i]);
            exit(EXIT_FAILURE);
        }
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Test PASSED\n");

    return 0;
}