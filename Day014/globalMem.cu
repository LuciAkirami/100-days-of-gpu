#include<stdio.h>
// #define DEBUG


// Helper function to check CUDA errors during execution
inline cudaError_t cudaCheck(cudaError_t result) {
    // Only check errors in debug mode
#if defined (DEBUG) || (_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Compilation Error %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);  // If error, assert fails
    }
#endif
    return result;
}

// Kernel to perform memory offset operation
__global__ void offset(float *a, int offset) {
    // Calculate global index for the element
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    // Increment the value at the corresponding memory location
    a[idx] = a[idx] + 1;
}

// Kernel to perform memory stride operation
__global__ void stride(float *a, int stride) {
    // Calculate global index with stride
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    // Increment the value at the corresponding memory location
    a[idx] = a[idx] + 1;
}

int main() {
    // Define the size of memory to be allocated in MB
    int numMegaBytes = 4;
    // Specify the device ID to be used for the GPU
    int deviceId = 0;

    // Calculate number of elements that fit in the allocated memory
    int numEle = numMegaBytes * 1024 * 1024 / sizeof(float);
    int size = numEle * sizeof(float);

    // Pointer for the allocated memory on the device
    float *a_d;
    // Allocate memory on the device (GPU) and check for errors
    cudaCheck(cudaMalloc(&a_d, size * 33));

    // Get device properties and display device name
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, deviceId));
    printf("Device Name: %s\n", prop.name);
    printf("Transfer Size (MB): %d\n", numMegaBytes);

    // Declare event variables for timing the GPU execution
    cudaEvent_t startEvent, stopEvent;
    float ms;

    // Create events to measure time
    cudaCheck(cudaEventCreate(&startEvent));
    cudaCheck(cudaEventCreate(&stopEvent));

    // Output the title for the bandwidth results
    printf("Offset, Bandwidth (GB/s)\n");

    // Set the block size and calculate the number of blocks for the grid
    int blockSize = 256;
    int numBlocks = ceil(numEle / blockSize);

    // Perform a warm-up run for offset kernel
    offset<<<numBlocks, blockSize>>>(a_d, 0);

    // Run the offset kernel for 33 different offsets and measure bandwidth
    for (int i = 0; i < 33; i++) {
        // Reset memory before each run
        cudaCheck(cudaMemset(a_d, 0, size));
        
        // Record the start event for timing
        cudaCheck(cudaEventRecord(startEvent, 0));
        // Launch the offset kernel with the current offset
        offset<<<numBlocks, blockSize>>>(a_d, i);
        // Record the stop event for timing
        cudaCheck(cudaEventRecord(stopEvent, 0));
        // Synchronize to ensure proper timing
        cudaCheck(cudaEventSynchronize(stopEvent));
        // Calculate elapsed time between the start and stop events
        cudaCheck(cudaEventElapsedTime(&ms, startEvent, stopEvent));

        // Output the offset and the corresponding bandwidth in GB/s
        printf(" %d, %.2f\n", i, 2 * numMegaBytes / ms);
    }

    // Output spacing before striding results
    printf("\n");
    printf("Striding\n");
    printf("Stride, Bandwidth (GB/s)\n");

    // Perform a warm-up run for stride kernel
    stride<<<numBlocks, blockSize>>>(a_d, 1);

    // Run the stride kernel for 33 different stride values and measure bandwidth
    for (int i = 0; i < 33; i++) {
        // Reset memory before each run
        cudaCheck(cudaMemset(a_d, 0, size));
        
        // Record the start event for timing
        cudaCheck(cudaEventRecord(startEvent, 0));
        // Launch the stride kernel with the current stride
        stride<<<numBlocks, blockSize>>>(a_d, i);
        // Record the stop event for timing
        cudaCheck(cudaEventRecord(stopEvent, 0));
        // Synchronize to ensure proper timing
        cudaCheck(cudaEventSynchronize(stopEvent));
        // Calculate elapsed time between the start and stop events
        cudaCheck(cudaEventElapsedTime(&ms, startEvent, stopEvent));

        // Output the stride and the corresponding bandwidth in GB/s
        printf("%d, %.2f\n", i, 2 * numMegaBytes / ms);
    }

    // Clean up events and free device memory
    cudaCheck(cudaEventDestroy(startEvent));
    cudaCheck(cudaEventDestroy(stopEvent));
    cudaFree(a_d);

    return 0;
}
