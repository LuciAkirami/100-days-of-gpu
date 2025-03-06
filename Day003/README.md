# Matrix-Vector Multiplication using CUDA

## Overview

This project demonstrates how to perform matrix-vector multiplication using both GPU (CUDA) and CPU implementations. The matrix-vector multiplication is a basic linear algebra operation, and this implementation shows how leveraging GPU parallelism can significantly speed up the computation when compared to traditional CPU execution.

### File: `matVecMul.cu`

The `matVecMul.cu` file contains the code for performing matrix-vector multiplication on both the CPU and GPU. The GPU implementation uses CUDA to parallelize the task, while the CPU implementation uses a straightforward sequential approach.

## Matrix-Vector Multiplication

Matrix-vector multiplication is an operation where:

Given:

-   A matrix \( A \) of size \( M \times M \)
-   A vector \( B \) of size \( M \times 1 \)

The result of multiplying the matrix \( A \) with vector \( B \) produces a new vector \( C \) of size \( M \times 1 \). The elements of \( C \) are computed as:

\[
C*i = \sum*{j=0}^{M-1} A\_{ij} \times B_j
\]

$\{ C_i = \sum_{j=0}^{M-1} A_{ij} \times B_j \}$

## Features

-   **GPU Implementation (CUDA)**: Utilizes CUDA to parallelize the matrix-vector multiplication. This approach allows each thread to compute one element of the result vector in parallel, significantly speeding up the operation for large matrices.
-   **CPU Implementation**: Provides a simple sequential implementation of matrix-vector multiplication for comparison with the GPU version.
-   **Execution Time Measurement**: The time taken for both GPU and CPU operations is measured and displayed to show the performance improvement with CUDA.

## Requirements

### Prerequisites

1. **CUDA Toolkit**: To compile and run the CUDA-based implementation, the CUDA Toolkit should be installed. It includes the necessary compiler and libraries to compile and run CUDA code.

    - **Installation guide**: https://developer.nvidia.com/cuda-toolkit

2. **C++ Compiler**: A C++ compiler (with C++11 or later) is required to compile the CUDA code.

3. **NVIDIA GPU**: To leverage the CUDA functionality, an NVIDIA GPU is required. However, the code will still compile and run on a system without a CUDA-capable GPU, but the GPU part will not be executed.

## Compilation

### Step 1: Install CUDA Toolkit

Ensure that the CUDA Toolkit is installed and your environment variables are set up correctly.

### Step 2: Compile the code

To compile the `matVecMul.cu` file, run the following command from the terminal:

```bash
nvcc -o matVecMul matVecMul.cu
```

This will generate an executable named `matVecMul` that you can run on your system.

### Step 3: Run the code

After compilation, run the program:

```bash
./matVecMul
```

The program will perform matrix-vector multiplication both on the GPU and the CPU, measuring the time taken for each operation and outputting the results to the console.

## Output

The program will print the following:

1. **GPU Execution Time**: The time taken for the GPU kernel to complete the matrix-vector multiplication.
2. **CPU Execution Time**: The time taken for the CPU to perform the same matrix-vector multiplication sequentially.

The program also contains a check for correctness, which ensures the results of both implementations match the expected output. (This check is currently commented out for performance benchmarking.)

Example output:

```
Time Taken in GPU (only Kernel): 0.25 seconds
Time Taken in CPU: 5.20 seconds
```

## Code Explanation

### 1. **Matrix Representation**

In this implementation, we use **single-row arrays** to represent the matrix. This means that the matrix is stored as a **1D array**, where each row of the matrix is stored contiguously in memory. For example:

-   For a matrix \( A \) of size \( M \times M \), we flatten it into a single-dimensional array `A_h` of size \( M^2 \).
-   The elements of the matrix can be accessed as if the matrix was stored in a 2D array using the following formula:

\[
A[i][j] = A_h[i \times M + j]
\]

Here, `A_h[i * M + j]` corresponds to the element at the \( i \)-th row and \( j \)-th column of the matrix.

This approach is efficient for memory storage and works well with CUDA, as it leverages the GPU's global memory in a linear fashion.

### 2. **CUDA Kernel for Matrix-Vector Multiplication**

```cpp
__global__
void matVecMulKernel(float *A, float *B, float *C, unsigned long int N, unsigned long int M) {
    unsigned long int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < M) {
        int sum = 0.0;
        for (int j = 0; j < M; j++) {
            sum += A[idx * M + j] * B[j];
        }
        C[idx] = sum;
    }
}
```

This is the CUDA kernel function that performs the matrix-vector multiplication. Each thread in the grid calculates one element of the output vector \( C \).

-   **Thread Index Calculation**: `idx` is the index of the thread within the grid. The number of threads per block is defined as 256, and the total number of blocks is calculated based on the size of the matrix \( N \) (i.e., the number of rows in the matrix).
-   **Matrix-Vector Multiplication**: Each thread calculates the value for a specific element in the resulting vector \( C \) by performing the dot product of a row in the matrix \( A \) and the vector \( B \).
-   **Matrix Access**: The formula `A[idx * M + j]` is used to access elements in the matrix \( A \), where `idx` corresponds to the row index, and `j` corresponds to the column index.

### 3. **GPU Function (`matvecmul`)**

```cpp
void matvecmul(float *A_h, float *B_h, float *C_h, unsigned long int size, unsigned long int size_out, unsigned long int N, unsigned long int M) {
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size_out);
    cudaMalloc((void **)&C_d, size_out);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_out, cudaMemcpyHostToDevice);

    dim3 blockDims(256, 1, 1);
    dim3 gridDims(ceil(N / float(blockDims.x)), 1, 1);

    auto start = chrono::high_resolution_clock::now();

    matVecMulKernel<<<gridDims, blockDims>>>(A_d, B_d, C_d, N, M);

    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    cout << "Time Taken in GPU (only Kernel): " << (float)duration.count()/1000 << " seconds" << endl;

    cudaMemcpy(C_h, C_d, size_out, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

-   **Memory Allocation**: This function allocates memory on the GPU for the matrices and the result vector.
-   **Data Transfer**: The input data (matrix \( A \) and vector \( B \)) are copied from the host to the device (GPU).
-   **Kernel Launch**: The `matVecMulKernel` is launched with a grid size and block size defined by `dim3`. The grid is sized to handle the total number of rows in the matrix \( A \).
-   **Time Measurement**: The time taken to execute the kernel is measured using `chrono::high_resolution_clock`.
-   **Data Transfer Back**: After the kernel finishes, the result vector \( C \) is copied back to the host.
-   **Memory Deallocation**: The memory allocated on the device is freed.

### 4. **CPU Function (`matVecMulCPU`)**

```cpp
void matVecMulCPU(float *A, float *B, float *C, unsigned long int N, unsigned long int M) {
    for (int i = 0; i < M; i++) {
        int sum = 0;
        for (int j = 0; j < M; j++) {
            sum += A[i * M + j] * B[j];
        }
        C[i] = sum;
    }
}
```

This function performs matrix-vector multiplication sequentially on the CPU. It loops through each row of the matrix \( A \) and calculates the dot product with the vector \( B \) to fill the result vector \( C \).

### 5. **Main Function**

```cpp
int main() {
    unsigned long int N = 100000000;
    unsigned long int M = 10000;
    unsigned long int size = N * sizeof(float);
    unsigned long int size_out = M * sizeof(float);

    float *A_h = (float *)malloc(size);
    float *B_h = (float *)malloc(size_out);
    float *C_h = (float *)malloc(size_out);

    for (int i = 0; i < N; i++) {
        if (i < M) {
            C_h[i] = 0;
            B_h[i] = 2;
        }
        A_h[i] = 1;
    }

    matvecmul(A_h, B_h, C_h, size, size_out, N, M);

    auto start = chrono::high_resolution_clock::now();
    matVecMulCPU(A_h, B_h, C_h, N, M);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    cout << "Time Taken in CPU: " << (float)duration.count()/1000 << " seconds" << endl;

    return 0;
}
```

In this section, the matrices are initialized, and the matrix-vector multiplication is performed both on the GPU and CPU. The execution time for both implementations is measured and printed.
