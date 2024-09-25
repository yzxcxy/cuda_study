#include <stdio.h>
#include "./tools/common.cuh"

int main(){
    //检测设备中GPU的数量
    int deviceCount=0;
    cudaError_t cuda_error;
    cuda_error=errorCheck(cudaGetDeviceCount(&deviceCount),__FILE__,__LINE__);

    //判断错误情况
    if(cuda_error!=cudaSuccess || deviceCount==0){
        printf("cudaGetDeviceCount failed!  Do you have a CUDA-Capable GPU installed?\n");
        exit(-1);
    }else{
        printf("CUDA Device Query...\n");
        printf("There are %d CUDA devices.\n",deviceCount);
    }

    //设置0号设备执行
    int index_device=1;
    cuda_error=errorCheck(cudaSetDevice(index_device),__FILE__,__LINE__);
    if(cuda_error!=cudaSuccess){
        printf("cudaSetDevice failed!  Do you have a CUDA-Capable GPU installed?\n");
        exit(-1);
    }else{
        printf("Set device %d to execute.\n",index_device);
    }

    return 0;
}