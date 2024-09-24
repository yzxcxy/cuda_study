#include<stdio.h>

int main(){
    //检测设备中GPU的数量
    int deviceCount=0;
    cudaError_t cuda_error=cudaGetDeviceCount(&deviceCount);

    //判断错误情况
    if(cuda_error!=cudaSuccess || deviceCount==0){
        printf("cudaGetDeviceCount failed!  Do you have a CUDA-Capable GPU installed?\n");
        exit(-1);
    }else{
        printf("CUDA Device Query...\n");
        printf("There are %d CUDA devices.\n",deviceCount);
    }

    //设置0号设备执行
    int index_device=0;
    cuda_error=cudaSetDevice(index_device);
    if(cuda_error!=cudaSuccess){
        printf("cudaSetDevice failed!  Do you have a CUDA-Capable GPU installed?\n");
        exit(-1);
    }else{
        printf("Set device %d to execute.\n",index_device);
    }
}