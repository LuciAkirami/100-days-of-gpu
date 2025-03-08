#include <stdio.h>

// CUDA kernel function for SAXPY operation: y = a * x + y
__global__
void saxpy(float *x, float *y, float a, int N){
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread index is within bounds
    if(idx < N){
        // Perform SAXPY operation for each element in parallel
        y[idx] = a * x[idx] + y[idx];
    }
}

int main(){
    float *x, *y, *x_d, *y_d;  // Host pointers for input/output arrays and device pointers for GPU memory
    int N = 20 * 1 << 20; // Size of arrays: 1 Million elements
    int size = N * sizeof(float);  // Total memory size for the arrays

    // Allocate memory on the host for x and y arrays
    x = (float *)malloc(size);
    y = (float *)malloc(size);

    // Initialize the host arrays with values
    for(int i=0; i<N; i++){
        x[i] = 1.0f;  // Initialize x to 1
        y[i] = 2.0f;  // Initialize y to 2
    }

    // Number of threads per block in a CUDA kernel launch
    int numThreadsPerBlock = 512;
    // Calculate number of blocks needed to cover the entire dataset
    int numBlocks = ceil(N / (float)numThreadsPerBlock);

    // Create CUDA events for measuring execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);  // Create start event
    cudaEventCreate(&stop);   // Create stop event

    // Allocate memory on the GPU for the x and y arrays
    cudaMalloc((void **)&x_d, size);
    cudaMalloc((void **)&y_d, size);

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    // Record the start time using CUDA events
    cudaEventRecord(start);

    // Launch the kernel with a specified number of blocks and threads per block
    saxpy<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, 2.0f, N);

    // Record the stop time using CUDA events
    cudaEventRecord(stop);

    // Copy the result back from device to host
    cudaMemcpy(y, y_d, size, cudaMemcpyDeviceToHost);

    // Synchronize the stop event to ensure all operations have finished
    // Waits until the completion of all work currently captured in this event. 
    // This prevents the CPU thread from proceeding until the event completes.
    // So it basically blocks until GPU performs all the operations
    cudaEventSynchronize(stop);

    // Calculate the elapsed time between start and stop events
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the time taken for GPU kernel execution
    printf("Time taken for GPU Kernel is %.2f milliseconds\n", milliseconds);

    // Check for errors in the results (SAXPY operation: expected y[i] = 4.0)
    float maxError = 0.0f;
    for(int i=0; i<N; i++){
        maxError = max(maxError, abs(y[i] - 4.0f));  // Find the maximum error
    }

    // Print the maximum error
    printf("Max Error is %.2f\n", maxError);
    // printf("%d \n",size);

    // N - Number of elements
    // N * 4 - Size of all elements in Bytes. As each element is a float and float takes 32bits aka  4 bytes
    // (N * 4) * 3 - Here we perform 2 read operations (x to x_d and y to y_d) and one write (y_d to y), hence
    // total operations performed are (2+1) = 3 and all of them have same number of elements N hence same memory N*4
    // (N*4*3)/milliseconds - By dividing total bytes by total time, we get time taken for each byte transfer in Bytes/ms
    // N*4*3/milliseconds/1e6 - By dividing this by 1e6, we transform Bytes/ms to GigaBytes/sec
    printf("Effective Bandwidth (GB/s): %f \n", N*4*3/milliseconds/1e6);

    // Free device and host memory
    cudaFree(x_d);
    cudaFree(y_d);
    free(x);
    free(y);
}
