## Querying Device Properties

```cpp
#include <stdio.h>

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
```

This code uses the function `cudaGetDeviceCount()` which returns in the argument nDevices the number of CUDA-capable devices attached to this system. Then in a loop we calculate the theoretical peak bandwidth for each device.The body of the loop uses `cudaGetDeviceProperties()` to populate the fields of the variable prop, which is an instance of the struct `cudaDeviceProp`. The program uses only three of `cudaDeviceProp`'s many members: `name`, `memoryClockRate`, and `memoryBusWidth`.

Running this on an NVIDIA RTX 3060 produced the following results

```
Device Number: 0
    Device Name: NVIDIA GeForce RTX 3060
    Memory Clock Rate(KHz): 7501000
    Memory Bus Width(bits): 192
    Peak Memory Bandwidth(GB/s): 360.048000
```

The peak bandwidth is similar to the calculation that was done in the README.md

## Compute Capability

The two important fields of `cudaDeviceProp` here are, major and minor. These describe the compute capability of the device, which is typically given in major.minor format and indicates the architecture generation. The first CUDA-capable device in the Tesla product line was the Tesla C870, which has a compute capability of 1.0. The first double-precision capable GPUs, such as Tesla C1060, have compute capability 1.3. GPUs of the Fermi architecture, such as the Tesla C2050 used above, have compute capabilities of 2.x, and GPUs of the Kepler architecture have compute capabilities of 3.x. Many limits related to the execution configuration vary with compute capability, as shown in the following table.

| Feature                      | Tesla C870 | Tesla C1060 | Tesla C2050 | Tesla K10 | Tesla K20 |
| ---------------------------- | ---------- | ----------- | ----------- | --------- | --------- |
| Compute Capability           | 1.0        | 1.3         | 2.0         | 3.0       | 3.5       |
| Max Threads per Thread Block | 512        | 512         | 1024        | 1024      | 1024      |
| Max Threads per SM           | 768        | 1024        | 1536        | 2048      | 2048      |
| Max Thread Blocks per SM     | 8          | 8           | 8           | 16        | 16        |

The grouping of threads into thread blocks mimics how thread processors are grouped on the GPU. This group of thread processors is called a streaming multiprocessor, denoted SM in the table above. The CUDA execution model issues thread blocks on multiprocessors, and once issued they do not migrate to other SMs.

Multiple thread blocks can concurrently reside on a multiprocessor subject to available resources (on-chip registers and shared memory) and the limit shown in the last row of the table. The limits on threads and thread blocks in this table are associated with the compute capability and not just a particular device: all devices of the same compute capability have the same limits. There are other characteristics, however, such as the number of multiprocessors per device, that depend on the particular device and not the compute capability. All of these characteristics, whether defined by the particular device or its compute capability, can be obtained using the `cudaDeviceProp` type.

You can generate code for a specific compute capability by using the nvcc compiler option -arch=sm_xx, where xx indicates the compute capability (without the decimal point). To see a list of compute capabilities for which a particular version of nvcc can generate code, along with other CUDA-related compiler options, issue the command nvcc --help and refer to the -arch entry.

When you specify an execution configuration for a kernel, keep in mind (and query at run time) the limits in the table above. This is especially important for the second execution configuration parameter: the number of threads per thread block. If you specify too few threads per block, then the limit on thread blocks per multiprocessor will limit the amount of parallelism that can be achieved. If you specify too many threads per thread block, well, that brings us to the next section.

## Handling CUDA Errors

All CUDA C Runtime API functions have a return value which can be used to check for errors that occur during their execution. In the example above, we can check for successful completion of `cudaGetDeviceCount()` like this:

```cpp
cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
```

We check to make sure `cudaGetDeviceCount()` returns the value `cudaSuccess`. If there is an error, then we call the function `cudaGetErrorString()` to get a character string describing the error.

Handling kernel errors is a bit more complicated because kernels execute asynchronously with respect to the host. To aid in error checking kernel execution, as well as other asynchronous operations, the CUDA runtime maintains an error variable that is overwritten each time an error occurs. The function `cudaPeekAtLastError()` returns the value of this variable, and the function `cudaGetLastError()` returns the value of this variable and also resets it to cudaSuccess.

We can check for errors in the saxpy kernel as follows.

```cpp
saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
cudaError_t errSync  = cudaGetLastError();
cudaError_t errAsync = cudaDeviceSynchronize();
if (errSync != cudaSuccess)
  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
if (errAsync != cudaSuccess)
  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
```

This code checks for both synchronous and asynchronous errors. Invalid execution configuration parameters, e.g. too many threads per thread block, are reflected in the value of `errSync` returned by `cudaGetLastError()`. Asynchronous errors that occur on the device after control is returned to the host, such as out-of-bounds memory accesses, require a synchronization mechanism such as `cudaDeviceSynchronize()`, which blocks the host thread until all previously issued commands have completed. Any asynchronous error is returned by `cudaDeviceSynchronize()`. We can also check for asynchronous errors and reset the runtime error state by modifying the last statement to call `cudaGetLastError()`.

```
if (errAsync != cudaSuccess)
  printf("Async kernel error: %s\n", cudaGetErrorString(cudaGetLastError());
```

Device synchronization is expensive, because it causes the entire device to wait, destroying any potential for concurrency at that point in your program. So use it with care. Typically, we can use preprocessor macros to insert asynchronous error checking only in debug builds of my code, and not in release builds.
