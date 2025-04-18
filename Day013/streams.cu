#include<stdio.h> // Include standard I/O library for printing

const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main()
{
    const int num_streams = 8;
    float ms;

    cudaEvent_t startEvent, stopEvent;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    cudaStream_t streams[num_streams];
    float *data[num_streams];

    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    printf("Elapsed Time: %f\n",ms);
    cudaDeviceReset();

    return 0;
}