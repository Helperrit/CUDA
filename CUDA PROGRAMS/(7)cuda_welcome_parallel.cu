#include <stdio.h>
#include <cuda.h>

// Kernel function
__global__ void printWelcomeMessage(int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index

    if (idx < N) {
        printf("Welcome to Parallel Programming from thread %d\n", idx);
    }
}

int main() {
    int N = 10;               // Number of times to print the message
    int threadsPerBlock = 4;  // Number of threads per block

    // Calculate the number of blocks needed
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    printWelcomeMessage<<<blocksPerGrid, threadsPerBlock>>>(N);

    // Synchronize to ensure kernel execution completes
    cudaDeviceSynchronize();

    return 0;
}
