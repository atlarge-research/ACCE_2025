#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ================== Two Kernel Implementations ==================
// Grid-stride version
__global__ void vectorAdd_grid_stride(const int *A, const int *B, int *C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < numElements; idx += stride) {
        C[idx] = A[idx] + B[idx];
    }
}

// Traditional single-thread version
__global__ void vectorAdd_no_stride(const int *A, const int *B, int *C, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        C[idx] = A[idx] + B[idx];
    }
}

// ================== Performance Testing Framework ==================
int main() {
    const int numElements = 1 << 20;  // Fixed problem size
    size_t size = numElements * sizeof(int);
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    // Configure test parameters
    const int test_dims[] = {16, 32, 64, 128, 256};
    const int num_tests = sizeof(test_dims)/sizeof(test_dims[0]);

    // Initialize host memory
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
        h_B[i] = i;
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Create CUDA event timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("| blk_dim | grid_dim (stride) | grid_dim (no-stride) | time_stride (ms) | time_no_stride (ms) |\n");
    printf("|---------|-------------------|----------------------|------------------|---------------------|\n");

    // Warm up
    vectorAdd_grid_stride<<<numSMs * 8 * 256 / 128, 128>>>(d_A, d_B, d_C, numElements);
    vectorAdd_no_stride<<<(numElements + 128 - 1) / 128, 128>>>(d_A, d_B, d_C, numElements);
    cudaDeviceSynchronize();

    for (int i = 0; i < num_tests; ++i) {
        const int blk_dim = test_dims[i];
        
        // Calculate grid dimensions for both versions
        const int grid_stride = numSMs * 8 * 256 / blk_dim;
        const int grid_no_stride = (numElements + blk_dim - 1) / blk_dim;

        // Test grid-stride version
        float time_stride;
        cudaMemset(d_C, 0, size);
        cudaEventRecord(start);
        vectorAdd_grid_stride<<<grid_stride, blk_dim>>>(d_A, d_B, d_C, numElements);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_stride, start, stop);

        // Test no-stride version
        float time_no_stride;
        cudaMemset(d_C, 0, size);
        cudaEventRecord(start);
        vectorAdd_no_stride<<<grid_no_stride, blk_dim>>>(d_A, d_B, d_C, numElements);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_no_stride, start, stop);

        printf("| %7d | %17d | %20d | %16.3f | %19.3f |\n",
               blk_dim, grid_stride, grid_no_stride, time_stride, time_no_stride);
    }

    // Clean up resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}