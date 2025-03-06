#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// CUDA Kernel to perform matrix-vector multiplication
__global__
void matVecMulKernel(float *A, float *B, float *C, unsigned long int N, unsigned long int M) {
    // Calculate the global index of the current thread
    unsigned long int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if the index is within the bounds of the output vector size
    if (idx < M) {
        int sum = 0.0;
        // Perform the matrix-vector multiplication for the current row
        for (int j = 0; j < M; j++) {
            sum += A[idx * M + j] * B[j];
        }
        // Store the result in the output vector C
        C[idx] = sum;
    }
}

// Function to perform matrix-vector multiplication on the GPU
void matvecmul(float *A_h, float *B_h, float *C_h, unsigned long int size, unsigned long int size_out, unsigned long int N, unsigned long int M) {
    // Declare device pointers for matrices A, B, and C
    float *A_d, *B_d, *C_d;
    
    // Allocate memory on the GPU
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size_out);
    cudaMalloc((void **)&C_d, size_out);

    // Copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_out, cudaMemcpyHostToDevice);

    // Define block and grid dimensions for kernel launch
    dim3 blockDims(256, 1, 1); // Number of threads per block (256 threads per block)
    dim3 gridDims(ceil(N / float(blockDims.x)), 1, 1); // Number of blocks in the grid, ceil ensures we cover all rows

    // Record start time for GPU kernel execution
    auto start = chrono::high_resolution_clock::now();

    // Launch the matrix-vector multiplication kernel on the GPU
    matVecMulKernel<<<gridDims, blockDims>>>(A_d, B_d, C_d, N, M);
    
    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();
    
    // Record end time for GPU kernel execution
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    // Output time taken for Matrix-Vector multiplication execution on the GPU
    cout << "Time Taken in GPU (only Kernel): " << (float)duration.count()/1000 << " seconds" << endl;

    // Copy the result back to the host from device
    cudaMemcpy(C_h, C_d, size_out, cudaMemcpyDeviceToHost);
    
    // Free the allocated device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

// CPU function to perform matrix-vector multiplication (for comparison)
void matVecMulCPU(float *A, float *B, float *C, unsigned long int N, unsigned long int M) {
    // Loop through each row of the matrix
    for (int i = 0; i < M; i++) {
        int sum = 0;
        // Perform the matrix-vector multiplication for the current row
        for (int j = 0; j < M; j++) {
            sum += A[i * M + j] * B[j];
        }
        // Store the result in the output vector C
        C[i] = sum;
    }
}

int main() {
    unsigned long int N = 100000000; // 10000x10000 matrix -> MxM
    unsigned long int M = 10000; // 10000X1 matrix -> Mx1
    unsigned long int size = N * sizeof(float); // Size of matrix A
    unsigned long int size_out = M * sizeof(float); // Size of output vector C

    // Allocate memory for host matrices A, B, and output vector C
    float *A_h = (float *)malloc(size); // 2D 10000x10000 matrix linearized to 1D
    float *B_h = (float *)malloc(size_out); // 1D Vector of size 10000
    float *C_h = (float *)malloc(size_out);

    // Initialize matrices A and B on the host
    for (int i = 0; i < N; i++) {
        if (i < M) {
            C_h[i] = 0; // Initialize the output vector C to 0
            B_h[i] = 2; // Initialize vector B to all 2's
        }
        A_h[i] = 1; // Initialize matrix A to all 1's
    }

    // Call the GPU matrix-vector multiplication function
    matvecmul(A_h, B_h, C_h, size, size_out, N, M);

    // Optionally, check if results are correct 
    // for (int i = 0; i < M; i++) {
    //     if (C_h[i] != 20000) {
    //         cout << i << endl; // Identify any incorrect results
    //     }
    // }

    // Record start time for CPU matrix-vector multiplication execution
    auto start = chrono::high_resolution_clock::now();

    // Call the CPU matrix-vector multiplication function for comparison
    matVecMulCPU(A_h, B_h, C_h, N, M);

    // Record end time for CPU matrix-vector multiplication execution
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    // Output time taken for Matrix-Vector multiplication execution on the CPU
    cout << "Time Taken in CPU: " << (float)duration.count()/1000 << " seconds" << endl;
    
    // Free the Memory
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}
