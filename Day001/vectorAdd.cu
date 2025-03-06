#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

void vecAddCPU(float *A, float *B, float *C, int n) {
    // CPU vector addition
    for (int i=0; i < n; i++) {
        C[i] = A[i] + B[i];  // Add the corresponding elements from A and B, store result in C
    }
}

__global__  
void vecAddKernel(float *A, float *B, float *C, int n) {
    // Kernel function for vector addition, executed on the GPU
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global index of the thread

    if (globalIdx < n) {  // Ensure index is within array bounds
        C[globalIdx] = A[globalIdx] + B[globalIdx];  // Add A[i] and B[i] and store in C[i]
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
    // Function to handle GPU memory allocation, data transfer, and kernel execution
    float *A_d, *B_d, *C_d;  // Declare pointers for device memory
    int size = n * sizeof(float);  // Calculate memory size for n elements

    cudaMalloc((void **)&A_d, size);  // Allocate memory on GPU for A
    cudaMalloc((void **)&B_d, size);  // Allocate memory on GPU for B
    cudaMalloc((void **)&C_d, size);  // Allocate memory on GPU for C

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);  
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = ceil(n / threadsPerBlock);

    auto start = chrono::high_resolution_clock::now();  // Start timer to measure kernel execution time
    vecAddKernel <<<blocksPerGrid, threadsPerBlock>>> (A_d, B_d, C_d, n);  // Launch the kernel with n/256 blocks, 256 threads per block
    cudaDeviceSynchronize();  // Ensure the kernel finishes before continuing
    auto end = chrono::high_resolution_clock::now();  // End timer
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);  // Measure the elapsed time in milliseconds

    // Output the time taken to execute the kernel
    cout << "Time Taken on GPU(Only Kernel): " << duration.count() << " milliseconds" << endl;

    // Copy result from device to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N = 1 << 30;  // Define N as 2^30, or 1 billion (a large enough array size for testing)

    // Allocate memory on the host (CPU)
    float *A_h = (float *)malloc(N * sizeof(float));  // Array A of size N
    float *B_h = (float *)malloc(N * sizeof(float));  // Array B of size N
    float *C_h = (float *)malloc(N * sizeof(float));  // Array C of size N to store the result

    // Initialize arrays A and B with random values
    for (int i = 0; i < N; i++) {
        A_h[i] = rand();  // Random value for A[i]
        B_h[i] = rand();  // Random value for B[i]
    }

    // Measure time taken for CPU-based vector addition
    auto start = chrono::high_resolution_clock::now();
    vecAddCPU(A_h, B_h, C_h, N);  // Perform vector addition on the CPU
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);  // Get elapsed time

    // Output time taken on the CPU
    cout << "Time Taken on CPU: " << duration.count() << " milliseconds" << endl;

    // Measure time taken for GPU-based vector addition
    vecAdd(A_h, B_h, C_h, N);  // Perform vector addition on the GPU

    return 0;  // End of program
}
