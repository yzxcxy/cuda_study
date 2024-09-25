#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>
#include "./tools/common.cuh"


__device__ int d_x = 1;
__device__ int d_y[2];

__global__ void kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;

    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}



int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    errorCheck(cudaGetDeviceProperties(&deviceProps, devID),__FILE__,__LINE__);
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    int h_y[2] = {10, 20};
    errorCheck(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2),__FILE__,__LINE__);

    dim3 block(2);
    dim3 grid(2);
    kernel<<<grid, block>>>();
    errorCheck(cudaDeviceSynchronize(),__FILE__,__LINE__);
    errorCheck(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2),__FILE__,__LINE__);
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);

    errorCheck(cudaDeviceReset(),__FILE__,__LINE__);

    return 0;
}