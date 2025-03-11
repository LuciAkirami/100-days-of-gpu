#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// Kernel function to apply leakyReLU on GPU
__global__
void leakyReluKernel(float *A, unsigned long int N){
    // Get the global thread index
    unsigned long int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Apply leakyReLU if the index is within bounds
    if (globalIdx < N) {
        // Leaky ReLU: If the value is less than 0, set it to 0.01 * x
        if (A[globalIdx] < 0.0) {
            A[globalIdx] = 0.01 * A[globalIdx];
        }
    }
}

// Kernel function to compute backward pass of leakyReLU on GPU (using A for gradients)
__global__
void leakyReluBackwardKernel(float *A, unsigned long int N){
    // Get the global thread index
    unsigned long int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute gradient for leakyReLU if the index is within bounds
    if (globalIdx < N) {
        // Gradient is 1 if A[globalIdx] >= 0, else 0.01 (leaky slope)
        if (A[globalIdx] < 0.0) {
            A[globalIdx] = 0.01;
        } else {
            A[globalIdx] = 1.0; // Gradient is 1 for positive values
        }
    }
}

// Function to perform leakyReLU on the GPU
void leakyReluGPU(float *A_h, unsigned int N){
    float *A_d;
    unsigned long int size = N * sizeof(float);

    // Allocate memory on GPU for array A
    cudaMalloc((void **)&A_d, size);

    // Copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

    // Configure number of threads per block and number of blocks
    int numThreadsPerBlock = 1024;
    int numBlocks = ceil(N / numThreadsPerBlock);

    // Record start time for GPU kernel execution
    auto start = chrono::high_resolution_clock::now();
    
    // Launch the LeakyReLU kernel on the GPU
    leakyReluKernel<<<numBlocks, numThreadsPerBlock>>>(A_d, N);
    // Wait for kernel to finish execution
    cudaDeviceSynchronize();
    
    // Record end time for GPU kernel execution
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    // Output time taken for LeakyReLU execution on GPU
    cout << "Time Taken in GPU (only Kernel): " << (float)duration.count()/1000 << " seconds" << endl;

    // Copy result back from device to host
    cudaMemcpy(A_h, A_d, size, cudaMemcpyDeviceToHost);
    // Free the allocated memory on the GPU
    cudaFree(A_d);
}

// Function to perform leakyReLU backward pass on the GPU (using A for gradients)
void leakyReluBackwardGPU(float *A_h, unsigned int N){
    float *A_d;
    unsigned long int size = N * sizeof(float);

    // Allocate memory on GPU for array A (used for gradient calculation as well)
    cudaMalloc((void **)&A_d, size);

    // Copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

    // Configure number of threads per block and number of blocks
    int numThreadsPerBlock = 1024;
    int numBlocks = ceil(N / numThreadsPerBlock);

    // Launch the backward kernel for leakyReLU on the GPU
    leakyReluBackwardKernel<<<numBlocks, numThreadsPerBlock>>>(A_d, N); // Using A_d to store gradients in place
    // Wait for kernel to finish execution
    cudaDeviceSynchronize();

    // Copy the result (gradients) back from device to host (array A is now updated)
    cudaMemcpy(A_h, A_d, size, cudaMemcpyDeviceToHost);

    // Free the allocated memory on the GPU
    cudaFree(A_d);
}

// Function to perform leakyReLU on the CPU
void leakyReluCPU(float *B, unsigned long int N){
    // Record start time for CPU computation
    auto start = chrono::high_resolution_clock::now();

    // Perform LeakyReLU operation for each element in the array
    for(int i = 0; i < N; i++) {
        if (B[i] < 0.0) {
            B[i] = 0.01 * B[i];  // Leaky slope for negative values
        }
    }

    // Record end time for CPU computation
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    // Output time taken for LeakyReLU execution on CPU
    cout << "Time Taken by CPU: " << (float)duration.count()/1000 << " seconds" << endl;
}

// Function to compute the backward pass of leakyReLU on the CPU (using A for gradients)
void leakyReluBackwardCPU(float *A, unsigned long int N){
    // Perform LeakyReLU backward pass for each element in the array
    for(int i = 0; i < N; i++) {
        if (A[i] < 0.0) {
            A[i] = 0.01;  // Gradient is 0.01 for negative values
        } else {
            A[i] = 1.0;   // Gradient is 1 for positive values
        }
    }
}

int main() {
    unsigned long int N = 1 << 30; // Set the size of the array (1 billion elements)
    unsigned long int size = N * sizeof(float);

    // Allocate memory for arrays on the host (CPU)
    float *A_h = (float *)malloc(size);
    
    // Initialize arrays with -1.0 (which will be changed by LeakyReLU)
    for(int i = 0; i < N; i++) {
        A_h[i] = -1.0; // Initial value for both forward and backward passes
    }

    // Perform LeakyReLU on the GPU and CPU
    leakyReluGPU(A_h, N);
    leakyReluCPU(A_h, N);

    // Perform backward pass for LeakyReLU on the GPU and CPU (using same array for gradients)
    leakyReluBackwardGPU(A_h, N);
    leakyReluBackwardCPU(A_h, N);

    // Free host memory
    free(A_h);

    return 0;
}
