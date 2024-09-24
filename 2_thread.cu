#include<stdio.h>

__global__ void print_thread_index() {
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int id=block_id*blockDim.x+thread_id;

    printf("blockIdx.x: %d, threadIdx.x: %d, id: %d\n", block_id, thread_id, id);
}


int main() {
    print_thread_index<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}