# ReLU Benchmarking on GPU and CPU

This project compares the performance of the Rectified Linear Unit (ReLU) operation on a large array using both GPU (CUDA) and CPU. The ReLU function replaces any negative values in an array with 0 and leaves positive values unchanged.

## Features

-   **GPU Implementation:** Uses CUDA to perform the ReLU operation in parallel on the GPU.
-   **CPU Implementation:** Performs the same operation serially on the CPU for comparison.
-   **Performance Measurement:** Both GPU and CPU operations are timed to compare execution speed.

## Prerequisites

-   CUDA Toolkit installed and configured for GPU-based computation.
-   A compatible GPU for CUDA computations.
-   C++ compiler with CUDA support.
-   Basic knowledge of CUDA programming.

## Files

1. **Main Code:** The main C++ code that implements both the GPU and CPU versions of ReLU. It also benchmarks the performance for both implementations.
2. **CUDA Kernel:** The `reluKernel` function is a CUDA kernel that is executed in parallel across multiple threads on the GPU to perform the ReLU operation.

3. **CPU Version:** The `reluCPU` function processes the array sequentially on the CPU.

4. **Time Measurement:** The code measures the time taken to run both GPU and CPU implementations using the `chrono` library.

## How to Compile and Run

### 1. Compile the Code

Use the `nvcc` compiler to compile the code. Run the following command in the terminal:

```bash
nvcc -o simpleRELU simpleRELU.cu
```

### 2. Run the Program

After compilation, you can run the program with:

```bash
./simpleRELU
```

The program will initialize two large arrays (1 billion elements each), apply ReLU on both the CPU and the GPU, and print the time taken for each.

## Expected Output

The program will output the time taken for both the GPU and CPU computations in seconds:

```bash
Time Taken in GPU (only Kernel): X.XXXX seconds
Time Taken by CPU: Y.YYYY seconds
```

The GPU should generally outperform the CPU, especially for large datasets.

## Notes

-   The array size `N` is set to 1 billion (`1 << 30`). You can adjust this value based on the memory available on your machine.
-   The performance difference will be significant when using a powerful GPU versus a CPU.
-   Ensure your system has sufficient GPU memory to handle the allocated arrays.
