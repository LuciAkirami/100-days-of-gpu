#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 1024  // Size of the matrices (NxN)

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float value = 0;
        for (int k = 0; k < width; k++) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *A_h, *B_h, *C_h;  // Host matrices
    float *A_d, *B_d, *C_d;  // Device matrices

    // Allocate host memory
    A_h = (float*)malloc(size);
    B_h = (float*)malloc(size);
    C_h = (float*)malloc(size);

    // Initialize matrices A and B with some values
    for (int i = 0; i < N * N; i++) {
        A_h[i] = rand() % 10;
        B_h[i] = rand() % 10;
    }

    // Allocate device memory
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);  // Grid size to cover all elements

    // Launch the kernel
    matrixMultiply<<<gridSize, blockSize>>>(A_d, B_d, C_d, N);

    // Check for errors in kernel launch
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Print out the first few elements of the result matrix
    std::cout << "Result matrix C (first 5 elements): " << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << C_h[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    free(A_h);
    free(B_h);
    free(C_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
