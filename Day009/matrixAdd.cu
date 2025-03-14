#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel function to add two matrices
__global__ void matrixAdd(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (idx < N && idy < N) {
        int index = idy * N + idx;  // Linear index in the 2D matrix
        C[index] = A[index] + B[index];  // Add corresponding elements
    }
}

int main() {
    int N = 1024;  // Size of the matrix (N x N)
    int *A_h, *B_h, *C_h;  // Host matrices
    int *A_d, *B_d, *C_d;  // Device matrices

    size_t size = N * N * sizeof(int);

    // Allocate memory for host matrices
    A_h = (int*)malloc(size);
    B_h = (int*)malloc(size);
    C_h = (int*)malloc(size);

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; i++) {
        A_h[i] = rand() % 100;
        B_h[i] = rand() % 100;
    }

    // Allocate memory for device matrices
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    // Launch the kernel
    matrixAdd<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N);

    // Check for kernel launch errors
    cudaDeviceSynchronize();

    // Copy the result matrix from device to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Print the first few elements of the result
    std::cout << "Result (first 10 elements): ";
    for (int i = 0; i < 10; i++) {
        std::cout << C_h[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // Free host memory
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}
