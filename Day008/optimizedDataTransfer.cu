#include<stdio.h>
#include <cuda_runtime.h>  // Include the CUDA runtime library
#include<assert.h>  // Include the assert library for debugging

// Inline function to check for CUDA runtime errors
inline
cudaError_t cudaCheck(cudaError_t result) {
    // Only check CUDA errors in DEBUG or _DEBUG mode
    #if defined(DEBUG) || defined(_DEBUG)
        if (result != cudaSuccess) {  // If CUDA error occurs
            fprintf(stderr, "Cuda Runtime Error %s\n", cudaGetErrorString(result));  // Print error message
            assert(result == cudaSuccess);  // Trigger an assertion failure if error exists
        }
    #endif
    return result;  // Return the result (for possible chaining)
}

// Function to profile memory transfer speeds (Host to Device, Device to Host)
void profileCopies(
    float *a,    // Pointer to the source array on the host
    float *b,    // Pointer to the destination array on the host
    float *c,    // Pointer to the destination array on the device
    unsigned int n,  // Number of elements in the array
    const char *desc   // Description for logging (e.g., "Pageable" or "Pinned")
) {
    printf("\n%s transfers:\n", desc);  // Print description of transfer type
    unsigned int bytes = n * sizeof(float);  // Calculate total number of bytes to transfer

    cudaEvent_t startEvent, stopEvent;  // CUDA events for timing

    cudaCheck( cudaEventCreate(&startEvent) );  // Create a CUDA event to mark the start time
    cudaCheck( cudaEventCreate(&stopEvent) );   // Create a CUDA event to mark the stop time

    // Host to Device transfer: Record start event, copy data, and record stop event
    cudaCheck( cudaEventRecord(startEvent, 0) );
    cudaCheck( cudaMemcpy(c, a, bytes, cudaMemcpyHostToDevice) );  // Copy from host to device
    cudaCheck( cudaEventRecord(stopEvent, 0) );
    cudaCheck( cudaEventSynchronize(stopEvent) );  // Wait for the stop event to complete

    float time;  // Variable to store elapsed time
    cudaCheck( cudaEventElapsedTime(&time, startEvent, stopEvent) );  // Calculate elapsed time in milliseconds

    // Calculate and print Host to Device bandwidth (in GB/s)
    printf("    Host to Device Bandwidth (GB/s) : %f\n", bytes * 1e-6 / time);

    // Device to Host transfer: Record start event, copy data, and record stop event
    cudaCheck( cudaEventRecord(startEvent, 0) );
    cudaCheck( cudaMemcpy(b, c, bytes, cudaMemcpyDeviceToHost) );  // Copy from device to host
    cudaCheck( cudaEventRecord(stopEvent, 0) );
    cudaCheck( cudaEventSynchronize(stopEvent) );  // Wait for the stop event to complete

    // Calculate and print Device to Host bandwidth (in GB/s)
    cudaCheck( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("    Device to Host Bandwidth (GB/s) : %f\n", bytes * 1e-6 / time);

    // Verify that the transfer was successful by comparing arrays
    for (unsigned int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("*** %s transfers failed ***\n", desc);  // If values don't match, print failure message
            break;
        }
    }

    // Clean up: Destroy the CUDA events
    cudaCheck( cudaEventDestroy(startEvent) );
    cudaCheck( cudaEventDestroy(stopEvent) );
}

int main() {
    unsigned int nElements = 4 * 1024 * 1024;  // Set number of elements (4 million elements)
    unsigned int sizeInBytes = nElements * sizeof(float);  // Calculate total size in bytes

    float *a_hPageable, *a_hPinned;
    float *b_hPageable, *b_hPinned;
    float *c_d;

    // Allocate memory for pageable arrays on the host
    a_hPageable = (float *)malloc(sizeInBytes);  // Regular malloc for pageable memory
    b_hPageable = (float *)malloc(sizeInBytes);

    // Allocate pinned (page-locked) memory on the host
    cudaCheck(cudaMallocHost((void **)&a_hPinned, sizeInBytes));  // Allocates pinned memory
    cudaCheck(cudaMallocHost((void **)&b_hPinned, sizeInBytes));
    
    // Allocate memory on the device
    cudaCheck(cudaMalloc((void **)&c_d, sizeInBytes));  // Device memory for transfers

    // Initialize source array (a_hPageable) with values 0 to nElements-1
    for (unsigned int i = 0; i < nElements; i++) a_hPageable[i] = i;

    // Copy the initialized pageable array to pinned memory
    memcpy(a_hPinned, a_hPageable, sizeInBytes);
    memset(b_hPageable, 0, sizeInBytes);  // Initialize destination arrays to 0
    memset(b_hPinned, 0, sizeInBytes);

    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));  // Get device properties for device 0

    // Print out the device name and transfer size
    printf("\nDevice: %s\n", prop.name);
    printf("Transfer Size (in MB): %d\n", sizeInBytes / (1024 * 1024));

    // Profile memory transfer speeds for pageable and pinned memory
    profileCopies(a_hPageable, b_hPageable, c_d, nElements, "Pageable");
    profileCopies(a_hPinned, b_hPinned, c_d, nElements, "Pinned");

    // Clean up: Free allocated memory
    cudaFree(c_d);  // Free device memory
    cudaFreeHost(a_hPinned);  // Free pinned host memory
    cudaFreeHost(b_hPinned);
    free(a_hPageable);  // Free regular host memory
    free(b_hPageable);
}
