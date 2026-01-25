/**
Basic CUDA Memory Copy and Kernel Execution Test
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

__global__ void simple_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 1.5f;
    }
}

int main() {
    printf("Basic CUDA Test\n");
    
    int n = 10;
    float h_data[10];
    float* d_data;
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    printf("GPU memory allocated\n");
    
    // Launch kernel
    int block_size = 32;
    int grid_size = (n + block_size - 1) / block_size;
    simple_kernel<<<grid_size, block_size>>>(d_data, n);
    
    // Check kernel error
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(kernel_err));
    } else {
        printf("Kernel launched successfully\n");
    }
    
    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results
    printf("Results: ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_data));
    
    return 0;
}
