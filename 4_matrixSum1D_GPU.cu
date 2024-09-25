#include <stdio.h>
#include "./tools/common.cuh"

__device__ float add(float a,float b){
    return a+b;
}

__global__ void add_from_GPU(float *A,float *B,float *C,int size){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<size){
        C[idx]=add(A[idx],B[idx]);
    }else{
        return;
    }
}

void initial_data(float *addr,int size){
    for(int i=0;i<size;i++){
        // 这里会将随机数限制在0-255之间，然后除以10，得到0-25.5之间的浮点数
        // 技巧，通过相与操作来限制数值大小范围
        addr[i]=(float)(rand()&0xFF)/10.0f;
    }
}

int main(){
    //1,设置GPU
    setGPU();
    
    //2.分配主机内存
    //1.1计算主机内存大小
    int elem_count=512;
    size_t bytes_count=elem_count*sizeof(float);

    //1.2定义三个指针，指向主机内存
    float *fp_host_A=(float *)malloc(bytes_count);
    float *fp_host_B=(float *)malloc(bytes_count);
    float *fp_host_C=(float *)malloc(bytes_count);
    
    if(fp_host_A==NULL||fp_host_B==NULL||fp_host_C==NULL){
        printf("malloc host memory failed\n");
        exit(-1);
    }else{
        printf("malloc host memory success\n");
        memset(fp_host_A,0,bytes_count);
        memset(fp_host_B,0,bytes_count);
        memset(fp_host_C,0,bytes_count);
    }

    //3.分配设备内存
    //3.1定义三个指针，指向设备内存
    float *fp_dev_A,*fp_dev_B,*fp_dev_C;
    cudaMalloc((float **)&fp_dev_A,bytes_count);
    cudaMalloc((float **)&fp_dev_B,bytes_count);
    cudaMalloc((float **)&fp_dev_C,bytes_count);

    if(fp_dev_A!=NULL&&fp_dev_B!=NULL&&fp_dev_C!=NULL){
        printf("malloc device memory success\n");
        //值设为0
        cudaMemset(fp_dev_A,0,bytes_count);
        cudaMemset(fp_dev_B,0,bytes_count);
        cudaMemset(fp_dev_C,0,bytes_count);
    }else{
        printf("malloc device memory failed\n");
        exit(-1);
    }

    //4.初始化主机内存
    srand(666);
    initial_data(fp_host_A,elem_count);
    initial_data(fp_host_B,elem_count);

    //5.将主机内存数据拷贝到设备内存
    cudaMemcpy(fp_dev_A,fp_host_A,bytes_count,cudaMemcpyHostToDevice);
    cudaMemcpy(fp_dev_B,fp_host_B,bytes_count,cudaMemcpyHostToDevice);
    cudaMemcpy(fp_dev_C,fp_host_C,bytes_count,cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((elem_count+block.x-1)/32);

    add_from_GPU<<<grid,block>>>(fp_dev_A,fp_dev_B,fp_dev_C,elem_count);
    cudaDeviceSynchronize();

    //6.将设备内存数据拷贝到主机内存(这里会有隐式的同步操作)
    cudaMemcpy(fp_host_C,fp_dev_C,bytes_count,cudaMemcpyDeviceToHost);

    //7.验证结果
    for(int i=0;i<10;i++){
        printf("fp_host_A[%d]=%f,fp_host_B[%d]=%f,fp_host_C[%d]=%f\n",i,fp_host_A[i],i,fp_host_B[i],i,fp_host_C[i]);
    }

    //8.释放内存
    free(fp_host_A);
    free(fp_host_B);
    free(fp_host_C);
    cudaFree(fp_dev_A);
    cudaFree(fp_dev_B);
    cudaFree(fp_dev_C);

    return 0;

}