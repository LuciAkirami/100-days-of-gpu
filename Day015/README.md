# 🔄 **Reversing an Array Using Shared Memory in CUDA**

This CUDA program illustrates how shared memory can be used to reverse the elements of a 64-element integer array efficiently using two types of shared memory allocations:

1. **Static Shared Memory**
2. **Dynamic Shared Memory**

Understanding shared memory and synchronization is essential for writing high-performance CUDA code, especially when optimizing for memory bandwidth and minimizing latency.

---

## 🚀 CUDA Kernels Overview

There are two main kernel functions in the code:

```cpp
__global__ void staticReverse(int *d, int n);
__global__ void dynamicReverse(int *d, int n);
```

Both perform the same logic:

-   Load an element from **global memory** into **shared memory**
-   Wait for all threads to finish (`__syncthreads()`)
-   Write the reversed version of the data back to **global memory**

---

## 📌 What is Shared Memory in CUDA?

Shared memory is a **low-latency**, **high-bandwidth**, on-chip memory space **shared among threads in the same block**. It's particularly useful for:

-   Staging data before reuse
-   Optimizing memory access patterns
-   Reducing global memory accesses

### Types of Shared Memory

| Type    | Declaration                  | Use case                           |
| ------- | ---------------------------- | ---------------------------------- |
| Static  | `__shared__ int s[64];`      | When array size is fixed           |
| Dynamic | `extern __shared__ int s[];` | When size is determined at runtime |

---

## 🧠 Static Shared Memory: Step-by-Step Breakdown

```cpp
__global__ void staticReverse(int *d, int n) {
  __shared__ int s[64];       // Statically declared shared memory
  int t = threadIdx.x;
  int tr = n - t - 1;
  s[t] = d[t];                // Load global to shared
  __syncthreads();            // Synchronize threads
  d[t] = s[tr];               // Write reversed value to global memory
}
```

### 🔁 Visual Representation of Thread Execution

**Before execution:**

```
Global Memory (d):   [ 0  1  2  3  ... 61 62 63 ]
ThreadIdx.x:         [ 0  1  2  3  ... 61 62 63 ]
```

**Step 1 – Load to Shared Memory (s[t] = d[t]):**

```
Shared Memory (s):   [ 0  1  2  3  ... 61 62 63 ]
```

**Step 2 – \_\_syncthreads():**

This function **blocks all threads** in the block until **every thread** reaches this point.

⛔ Without `__syncthreads()`, some threads might try to **read from shared memory before others have finished writing**, leading to undefined behavior and race conditions.

**Step 3 – Write back reversed value (d[t] = s[tr]):**

```
Global Memory (d):   [ 63 62 61 60 ... 2 1 0 ]
```

---

## ⚙️ Dynamic Shared Memory: More Flexible

```cpp
__global__ void dynamicReverse(int *d, int n) {
  extern __shared__ int s[];   // Size determined at runtime
  int t = threadIdx.x;
  int tr = n - t - 1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}
```

### 🧠 Dynamic Shared Memory Allocation at Launch

```cpp
dynamicReverse<<<1, 64, 64 * sizeof(int)>>>(d_d, n);
```

-   The third parameter (`64 * sizeof(int)`) tells CUDA to allocate **64 integers** in shared memory for this kernel invocation.

---

## ➕ What If You Need Multiple Shared Arrays?

CUDA only allows one **unsized extern array** in dynamic shared memory, but you can partition it manually.

```cpp
extern __shared__ int s[];                 // Base pointer

int *intData = s;                          // First nI integers
float *floatData = (float*)&intData[nI];   // Next nF floats
char *charData = (char*)&floatData[nF];    // Remaining bytes
```

### ✅ Launch with Combined Shared Memory Size:

```cpp
myKernel<<<gridSize, blockSize,
  nI * sizeof(int) + nF * sizeof(float) + nC * sizeof(char)>>>(...);
```

## 💡 Visualization Idea (Diagram)

Here’s a conceptual flow:

```
+------------------------+
|   Global Memory (d)    |
|   [0, 1, 2, ..., 63]   |
+------------------------+
           |
           | (copy into shared memory by threads)
           V
+------------------------+
| Shared Memory (s)      |
|   [0, 1, 2, ..., 63]   |
+------------------------+
           |
           | (reversed indexing)
           V
+------------------------+
|  Global Memory (d)     |
| [63, 62, ..., 0]       |
+------------------------+
```
