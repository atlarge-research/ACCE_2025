// occupancy_test.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd_grid_stride(const int *A, const int *B, int *C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < numElements; idx += stride) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vectorAdd_no_stride(const int *A, const int *B, int *C, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[]) {
    
    // Add at the beginning of the main function:
    if (argc == 2 && strcmp(argv[1], "get_num_sms") == 0) {
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        printf("%d", numSMs);
        return 0;
    }
    
    if (argc != 3) {
        printf("Usage: %s <block_dim> <kernel_name>\n", argv[0]);
        return 1;
    }

    const int numElements = 1 << 20;
    size_t size = numElements * sizeof(int);
    int *d_A, *d_B, *d_C;

    // Allocate memory only to trigger kernel execution
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Parse parameters
    int blk_dim = atoi(argv[1]);
    const char *kernel_name = argv[2];

    // Calculate grid dimension
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int grid_dim = (strcmp(kernel_name, "vectorAdd_grid_stride") == 0) ?
                   numSMs * 8 * 256 / blk_dim : (numElements + blk_dim - 1) / blk_dim;

    // Execute kernel (actual computation is not important, mainly to trigger execution for measurement)
    if (strcmp(kernel_name, "vectorAdd_grid_stride") == 0) {
        vectorAdd_grid_stride<<<grid_dim, blk_dim>>>(d_A, d_B, d_C, numElements);
    } else {
        vectorAdd_no_stride<<<grid_dim, blk_dim>>>(d_A, d_B, d_C, numElements);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}