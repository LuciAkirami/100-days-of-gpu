#include <stdio.h>

int main(){
    // Declare a variable to store the number of available CUDA devices
    int nDevices;

    // Get the number of available CUDA devices
    cudaGetDeviceCount(&nDevices);

    // Loop through each available device (from 0 to nDevices-1)
    for(int i=0; i<nDevices; i++) {
        // Declare a variable to store the properties of the current device
        // We are creating a prop struct
        cudaDeviceProp prop;

        // Get the properties of the device with index i and store it in 'prop'
        cudaGetDeviceProperties(&prop, i);

        // Print the device number
        printf("Device Number: %d\n", i);

        // Print the name of the device
        printf("    Device Name: %s\n", prop.name);

        // Print the memory clock rate of the device in KHz
        printf("    Memory Clock Rate(KHz): %d\n", prop.memoryClockRate);

        // Print the memory bus width of the device in bits
        printf("    Memory Bus Width(bits): %d\n", prop.memoryBusWidth);

        // Calculate and print the theoretical peak memory bandwidth in GB/s
        // Formula: Peak Memory Bandwidth = 2 * (Memory Bus Width / 8) * Memory Clock Rate (in GHz)
        printf("    Peak Memory Bandwidth(GB/s): %f\n",
        2.0 * (prop.memoryBusWidth / 8) * prop.memoryClockRate * 1e-6);
    }
}
