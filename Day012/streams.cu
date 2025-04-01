#include<stdio.h> // Include standard I/O library for printing

// Function to check CUDA errors
inline
cudaError_t cudaCheck(cudaError_t result){
#if defined(DEBUG) || defined(_DEBUG) // Check if DEBUG or _DEBUG is defined
    if (result != cudaSuccess) { // If the result indicates an error
        fprintf(stderr, "Cuda Runtime Error: %s\n", cudaGetErrorString(result)); // Print the error message
        assert(result == cudaSuccess); // Assert that there is no error (debugging purpose)
    }
#endif
    return result; // Return the result (either success or error)
}

// CUDA kernel to modify array elements
__global__
void mulKernel(float *a, int offset, int n) {
    int idx = offset + blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global index of the thread

    if (idx < n){ // If the thread index is within bounds
        a[idx] = a[idx] + 1.1; // Add 1.1 to the corresponding element in array 'a'
    }
}

// Function to calculate maximum error between expected and actual values
float max_error(float *a, int n){
    float maxError = 0.0; // Initialize maximum error to 0
    float error = 0.0; // Initialize error variable

    for (int i=0; i<n; i++){ // Loop through each element in the array
        // fabs is the absolute value function for float types
        error = fabs(a[i] - (i + 1.1)); // Calculate the absolute difference between actual and expected value
        if (error > maxError){ // If the error is greater than the current max error
            maxError = error; // Update max error
        }
    }

    return maxError; // Return the maximum error found
}

int main (){
    int n = 4 * 1024 * 1024; // Define the size of the array (4 million elements)
    int numStreams = 4; // Number of CUDA streams to be used
    int streamSize = n / numStreams; // Number of elements each stream will process
    int streamBytes = streamSize * sizeof(float); // Size of data each stream will transfer (in bytes)
    int bytes = n * sizeof(float); // Total size of data to be processed (in bytes)

    int deviceId = 0; // Set the device ID to 0 (first CUDA device)

    int blockSize = 256; // Set block size for CUDA kernel

    // Query device properties for the selected device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId); // Get the properties of the device
    cudaCheck( cudaSetDevice(deviceId) ); // Set the device to be used for CUDA operations

    // Print device and data processing information
    printf("Device Name: %s\n", prop.name);
    printf("Total Data Processed (in MB): %dMB\n", bytes / (1024 * 1024)); // Print total data size in MB
    printf("Number of elements Processed by each Stream: %d\n", streamSize); // Print number of streams
    printf("Data Processed per Stream (in MB): %dMB\n", streamBytes / (1024 * 1024)); // Print data processed per stream in MB

    // Allocate pinned (page-locked) memory for host data
    float *a_h, *a_d;
    cudaCheck( cudaMallocHost((void **)&a_h, bytes) ); // Allocate pinned memory for host array
    cudaCheck( cudaMalloc((void **)&a_d, bytes) ); // Allocate memory for device array

    float ms; // Variable to store time in milliseconds

    // Declare streams and events for timing
    cudaStream_t streams[numStreams]; // Declare an array of streams
    cudaEvent_t startEvent, stopEvent, dummyEvent; // Declare events for timing

    // Create events to track the timing of operations
    cudaCheck( cudaEventCreate(&startEvent) ); // Create start event
    cudaCheck( cudaEventCreate(&stopEvent) ); // Create stop event
    cudaCheck( cudaEventCreate(&dummyEvent) ); // Create dummy event (not used, but created for consistency)

    // Create CUDA streams
    for (int i = 0; i < numStreams; i++){
        cudaCheck( cudaStreamCreate(&streams[i]) ); // Create each stream
    }

    // Initialize the host array with dummy data (sequential integers)
    for (int i = 0; i < n; i++) a_h[i] = i;

    // Sequential data transfer and kernel execution (baseline)
    cudaCheck( cudaEventRecord(startEvent, 0) ); // Record start event
    cudaCheck( cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice) ); // Copy data from host to device
    mulKernel<<<n/blockSize, blockSize>>>(a_d, 0, n); // Launch kernel on the device
    cudaCheck( cudaMemcpy(a_h, a_d, bytes, cudaMemcpyDeviceToHost) ); // Copy data back from device to host
    cudaCheck( cudaEventRecord(stopEvent, 0) ); // Record stop event
    cudaCheck( cudaEventSynchronize(stopEvent) ); // Synchronize the stop event
    cudaCheck( cudaEventElapsedTime(&ms, startEvent, stopEvent) ); // Measure elapsed time in milliseconds

    // Print results for sequential execution
    printf("\nTime taken for Sequential Transfer and Execute is %fms\n", ms);
    printf("Maximum Error: %e\n", max_error(a_h, 10)); // Print maximum error

    // Reset the host array with new dummy data
    memset(a_h, 0, bytes); // Set all elements to 0
    for (int i = 0; i < n; i++) a_h[i] = i; // Fill array with sequential integers

    // Approach 1: Loop Over {Copy, Kernel, Copy}
    cudaCheck( cudaEventRecord(startEvent, 0) ); // Record start event
    for (int i = 0; i < numStreams; i++) { // Loop over the streams
        int offset = i * streamSize; // Calculate the offset for each stream
        cudaCheck( cudaMemcpyAsync(&a_d[offset], &a_h[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]) ); // Asynchronous copy from host to device
        mulKernel<<<streamSize/blockSize, blockSize, 0, streams[i]>>>(a_d, offset, n); // Launch kernel for each stream
        cudaCheck( cudaMemcpyAsync(&a_h[offset], &a_d[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]) ); // Asynchronous copy from device to host
    }
    cudaCheck( cudaEventRecord(stopEvent, 0) ); // Record stop event
    cudaCheck( cudaEventSynchronize(stopEvent) ); // Synchronize the stop event
    cudaCheck( cudaEventElapsedTime(&ms, startEvent, stopEvent) ); // Measure elapsed time in milliseconds

    // Print results for Approach 1
    printf("\nTime taken for Asynchronous Transfer and Execute using Streams - Approach 1: %f\n", ms);
    printf("Maximum Error: %e\n", max_error(a_h, 10)); // Print maximum error

    // Reset the host array with new dummy data
    memset(a_h, 0, bytes); // Set all elements to 0
    for (int i = 0; i < n; i++) a_h[i] = i; // Fill array with sequential integers

    // Approach 2: Loop over Copy, Loop over Kernel, Loop over Copy
    cudaCheck( cudaEventRecord(startEvent, 0) ); // Record start event
    for (int i = 0; i < numStreams; i++) { // Loop over streams for the copy phase
        int offset = i * streamSize; // Calculate the offset for each stream
        cudaCheck( cudaMemcpyAsync(&a_d[offset], &a_h[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]) ); // Asynchronous copy from host to device
    }

    for (int i = 0; i < numStreams; i++) { // Loop over streams for the kernel phase
        int offset = i * streamSize; // Calculate the offset for each stream
        mulKernel<<<streamSize/blockSize, blockSize, 0, streams[i]>>>(a_d, offset, n); // Launch kernel for each stream
    }

    for (int i = 0; i < numStreams; i++) { // Loop over streams for the copy back phase
        int offset = i * streamSize; // Calculate the offset for each stream
        cudaCheck( cudaMemcpyAsync(&a_h[offset], &a_d[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]) ); // Asynchronous copy from device to host
    }
    cudaCheck( cudaEventRecord(stopEvent, 0) ); // Record stop event
    cudaCheck( cudaEventSynchronize(stopEvent) ); // Synchronize the stop event
    cudaCheck( cudaEventElapsedTime(&ms, startEvent, stopEvent) ); // Measure elapsed time in milliseconds

    // Print results for Approach 2
    printf("\nTime taken for Asynchronous Transfer and Execute using Streams - Approach 2: %f\n", ms);
    printf("Maximum Error: %e\n", max_error(a_h, 10)); // Print maximum error

    return 0; // Return from the main function
}