### Detailed Explanation of Code

This C++ program performs vector addition both on the CPU and the GPU. It demonstrates how a simple operation (element-wise addition of two arrays) can be performed in parallel on the GPU, and compares its performance against a sequential CPU implementation. Below, we explain each section of the program in detail.

#### 1. **Includes and Namespace**

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
```

-   `#include <iostream>`: Includes the header file for input-output operations in C++.
-   `#include <cuda_runtime.h>`: Includes the CUDA runtime API, allowing interaction with GPU hardware for memory management, kernel launches, etc.
-   `#include <chrono>`: Includes the C++ standard library's time utilities, which will be used to measure the execution time of the CPU and GPU operations.

#### 2. **CPU Function (`vecAddCPU`)**

```cpp
void vecAddCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}
```

-   This function performs the vector addition on the CPU. It iterates over each element in the arrays `A` and `B`, adds them together, and stores the result in array `C`.

#### 3. **GPU Kernel Function (`vecAddKernel`)**

```cpp
__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < n) {
        C[globalIdx] = A[globalIdx] + B[globalIdx];
    }
}
```

-   `__global__` indicates that this function is a CUDA kernel, which will be executed by multiple threads in parallel on the GPU.
-   Each thread computes the addition for a specific index `globalIdx` (calculated using the block index and thread index).
-   The conditional `if (globalIdx < n)` ensures that only valid array indices are accessed, preventing out-of-bounds memory access.

#### 4. **GPU Vector Addition Function (`vecAdd`)**

```cpp
void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = ceil(n / threadsPerBlock);

    auto start = chrono::high_resolution_clock::now();
    vecAddKernel <<<blocksPerGrid, threadsPerBlock>>> (A_d, B_d, C_d, n);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    cout << "Time Taken on GPU(Only Kernel): " << duration.count() << " milliseconds" << endl;

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

-   This function handles all the operations related to running the GPU vector addition, including:
    -   **Memory Allocation:** Allocates memory on the GPU (`A_d`, `B_d`, and `C_d` for arrays `A`, `B`, and `C`).
    -   **Memory Copying:** Copies data from the host (CPU) arrays to device (GPU) memory using `cudaMemcpy`.
    -   **Kernel Launch:** Launches the kernel `vecAddKernel` on the GPU with a grid of blocks, each with 256 threads.
    -   **Synchronization:** Ensures the GPU finishes executing the kernel before proceeding.
    -   **Time Measurement:** Measures the execution time for the kernel.
    -   **Memory Cleanup:** Frees the allocated device memory.

#### 5. **Main Function**

```cpp
int main() {
    int N = 1 << 30;  // Set N to 2^30 (a large number for testing)
    float *A_h = (float *)malloc(N * sizeof(float));  // Allocate host memory for array A
    float *B_h = (float *)malloc(N * sizeof(float));  // Allocate host memory for array B
    float *C_h = (float *)malloc(N * sizeof(float));  // Allocate host memory for array C

    for (int i = 0; i < N; i++) {
        A_h[i] = rand();  // Initialize array A with random values
        B_h[i] = rand();  // Initialize array B with random values
    }

    // Measure time taken for CPU-based vector addition
    auto start = chrono::high_resolution_clock::now();
    vecAddCPU(A_h, B_h, C_h, N);  // Perform CPU-based vector addition
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    cout << "Time Taken on CPU: " << duration.count() << " milliseconds" << endl;

    // Measure time taken for GPU-based vector addition
    vecAdd(A_h, B_h, C_h, N);  // Perform GPU-based vector addition

    return 0;
}
```

-   **Initialization:** The program initializes the size `N` of the arrays and allocates memory for the input and output arrays on the host (CPU).
-   **CPU Operation:** It performs the vector addition on the CPU and measures the time taken.
-   **GPU Operation:** It then performs the same operation on the GPU and outputs the time taken for the kernel to execute.

#### Conclusion

This program compares the performance of a CPU-based vector addition with a GPU-based implementation using CUDA. By leveraging the parallel processing power of the GPU, the program achieves significant performance gains over the CPU-based version.

### vecAdd.cu

This contains the similar code as `vectorAdd.cu`. It contains both the CPU version of vector addition and GPU version of vector addition. Running this code will print the time taken to perform the operation in both the CPU and GPU
