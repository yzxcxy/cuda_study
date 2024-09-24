#include <stdio.h>


// __global__ is a CUDA specifier that indicates a function that runs on the device and can be called from host code.
__global__ void hello_from__gpu(){
    printf("Hello World from GPU!\n");
}

int main(){
    // <<<4,4>>> is a CUDA syntax that specifies the number of blocks and threads to run the kernel. This will be printed 16 times.
    hello_from__gpu<<<4,4>>>();
    // cudaDeviceSynchronize() is a CUDA function that waits for the device to finish its execution.
    cudaDeviceSynchronize();

    return 0;
}