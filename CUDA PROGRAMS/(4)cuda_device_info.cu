#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp deviceProp;
    int devID;

    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&deviceProp, devID);

    printf("CUDA Device Information:\n\n");
    printf("1. Warp size: %d\n", deviceProp.warpSize);
    printf("2. Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("3. Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("4. Maximum sizes of each dimension of a block: x = %d, y = %d, z = %d\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("5. Maximum sizes of each dimension of a grid: x = %d, y = %d, z = %d\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("6. Maximum memory pitch: %zu bytes\n", deviceProp.memPitch);

    return 0;
}