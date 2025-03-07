#include <stdio.h>

__global__
void saxpy(float *x, float *y, float a, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N){
        y[idx] = a * x[idx] + y[idx];
    }
}

int main(){
    float *x, *y, *x_d, *y_d;
    int N = 1 << 20; // 1 Million
    int size = N * sizeof(float);

    x = (float *)malloc(size);
    y = (float *)malloc(size);

    for(int i=0; i<N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int numThreadsPerBlock = 256;
    int numBlocks = ceil(N / numThreadsPerBlock);

    cudaMalloc((void **)&x_d, size);
    cudaMalloc((void **)&y_d, size);

    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    saxpy<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, 2.0f, N);

    cudaMemcpy(y, y_d, size, cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for(int i=0; i<N; i++){
        maxError = max(maxError, abs(y[i] - 4.0f));
    }

    printf("Max Error is %.2f\n",maxError);

    cudaFree(x_d);
    cudaFree(y_d);

    free(x);
    free(y);
}