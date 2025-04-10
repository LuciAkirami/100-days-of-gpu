#include <stdio.h>

// Kernel using static shared memory to reverse array
__global__ void staticReverseKernel(int *d, int n) {
    __shared__ int s[64];  // Declare static shared memory array of size 64

    int i = threadIdx.x;       // Thread index within the block
    int inv_i = n - i - 1;     // Compute reverse index

    s[i] = d[i];               // Load data from global to shared memory
    __syncthreads();           // Synchronize to make sure all data is loaded

    d[i] = s[inv_i];           // Write reversed data back to global memory
}

// Kernel using dynamic shared memory to reverse array
__global__ void dynamicReverseKernel(int *d, int n) {
    extern __shared__ int s[]; // Declare dynamic shared memory

    int i = threadIdx.x;
    int inv_i = n - i - 1;

    s[i] = d[i];               // Load data from global to shared memory
    __syncthreads();           // Synchronize threads before reading

    d[i] = s[inv_i];           // Write reversed data to global memory
}

int main() {
    int n = 64;
    int a[64], r[64], d[64];   // Host arrays: input (a), reference (r), and device result copy (d)

    // Initialize the input and reference arrays
    for (int i = 0; i < 64; i++) {
        a[i] = i;              // a = [0, 1, ..., 63]
        r[i] = n - i - 1;      // r = [63, 62, ..., 0]
        d[i] = 0;              // Initialize result array to zero
    }

    int *d_d;                  // Device pointer

    // Allocate device memory
    cudaMalloc(&d_d, n * sizeof(int));

    // ----------- Static Shared Memory Test ------------

    // Copy input array from host to device
    cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block of 64 threads using static shared memory
    staticReverseKernel<<<1, 64>>>(d_d, n);

    // Copy result back from device to host
    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results from static shared memory kernel
    for (int i = 0; i < n; i++) {
        if (r[i] != d[i]) {
            printf("Error r[%d] != d[%d] (%d, %d)\n", i, i, r[i], d[i]);
        }
    }

    // ----------- Dynamic Shared Memory Test ------------

    // Copy input array again from host to device
    cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block of 64 threads and dynamically allocated shared memory
    dynamicReverseKernel<<<1, 64, n * sizeof(int)>>>(d_d, n);

    // Copy result back from device to host
    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results from dynamic shared memory kernel
    for (int i = 0; i < n; i++) {
        if (r[i] != d[i]) {
            printf("Error r[%d] != d[%d] (%d, %d)\n", i, i, r[i], d[i]);
        }
    }

    // Free device memory
    cudaFree(d_d);
}