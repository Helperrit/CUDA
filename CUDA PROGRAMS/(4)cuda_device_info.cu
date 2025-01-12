#include <stdio.h>
#include <cuda.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get the number of CUDA-capable devices

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("\nDevice %d: %s\n", device, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %lu bytes\n", deviceProp.totalGlobalMem);
        printf("  Shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Registers per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("  Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Maximum grid dimensions: %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum block dimensions: %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Clock rate: %d kHz\n", deviceProp.clockRate);
        printf("  Memory clock rate: %d kHz\n", deviceProp.memoryClockRate);
        printf("  Memory bus width: %d bits\n", deviceProp.memoryBusWidth);
    }

    return 0;
}
