#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// Kernel function to apply ReLU on GPU
__global__
void reluKernel(float *A, unsigned long int N){
    // Get the global thread index
    unsigned long int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Apply ReLU if the index is within bounds
    if (globalIdx < N) {
        // ReLU: If the value is less than 0, set it to 0
        if (A[globalIdx] < 0.0) {
            A[globalIdx] = 0.0;
        }
    }
}

// Function to perform ReLU on the GPU
void reluGPU(float *A_h, unsigned int N){
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
    
    // Launch the ReLU kernel on the GPU
    reluKernel<<<numBlocks, numThreadsPerBlock>>>(A_d, N);
    // Wait for kernel to finish execution
    cudaDeviceSynchronize();
    
    // Record end time for GPU kernel execution
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    // Output time taken for ReLU execution on GPU
    cout << "Time Taken in GPU (only Kernel): " << (float)duration.count()/1000 << " seconds" << endl;

    // Copy result back from device to host
    cudaMemcpy(A_h, A_d, size, cudaMemcpyDeviceToHost);
    // Free the allocated memory on the GPU
    cudaFree(A_d);
}

// Function to perform ReLU on the CPU
void reluCPU(float *B, unsigned long int N){
    // Record start time for CPU computation
    auto start = chrono::high_resolution_clock::now();

    // Perform ReLU operation for each element in the array
    for(int i = 0; i < N; i++) {
        if (B[i] < 0.0) {
            B[i] = 0.0;
        }   
    }

    // Record end time for CPU computation
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    // Output time taken for ReLU execution on CPU
    cout << "Time Taken by CPU: " << (float)duration.count()/1000 << " seconds" << endl;
}

int main() {
    unsigned long int N = 1 << 30; // Set the size of the array (1 billion elements)
    unsigned long int size = N * sizeof(float);

    // Allocate memory for arrays on the host (CPU)
    float *A_h = (float *)malloc(size);
    float *B_h = (float *)malloc(size);

    // Initialize arrays with -1.0 (which will be changed by ReLU)
    for(int i = 0; i < N; i++) {
        A_h[i] = -1.0;
        B_h[i] = -1.0;     
    }

    // Perform ReLU on the GPU and CPU
    reluGPU(A_h, N); 
    reluCPU(B_h, N);

    // Free host memory
    free(A_h);
    free(B_h);

    return 0;
}