#include<iostream>
#include <stdlib.h> 
#include <time.h>
#include <sys/time.h>
using namespace std;

__global__ void addVec(int *ad,int*bd,int*cd){

    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int tidx=tx+bx*blockDim.x,tidy=ty+by*blockDim.y;
    cd[tidx*1024+tidy]=ad[tidx*1024+tidy]+bd[tidx*1024+tidy];
    // cd[tidy*1024+tidx]=ad[tidy*1024+tidx]+bd[tidy*1024+tidx];
    // if(bx==0,by==0){
    //     printf("thread id:%d\n",tx*32+ty);
    //     // printf("thread id:%d\n",ty*32+tx);
    // }
}

__global__ void addVec1(int *ad,int*bd,int*cd){

    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int tidx=tx+bx*blockDim.x,tidy=ty+by*blockDim.y;
    // cd[tidx*1024+tidy]=ad[tidx*1024+tidy]+bd[tidx*1024+tidy];
    cd[tidy*1024+tidx]=ad[tidy*1024+tidx]+bd[tidy*1024+tidx];
    // if(bx==0,by==0){
    //     printf("thread id:%d\n",tx*32+ty);
    //     // printf("thread id:%d\n",ty*32+tx);
    // }
}
int main(){
    int *a,*b,*c,*ad,*bd,*cd;
    int len=1024*1024;
    int total_size=len*sizeof(int);
    a=(int*)malloc(total_size);
    b=(int*)malloc(total_size);
    c=(int*)malloc(total_size);

    for(int i=0;i<len;i++){
        a[i]=1;
        b[i]=3;
    }
    cudaMalloc((void**)&ad,total_size);
    cudaMalloc((void**)&bd,total_size);
    cudaMalloc((void**)&cd,total_size);
    cudaMemcpy(ad,a,total_size,cudaMemcpyHostToDevice);
    cudaMemcpy(bd,b,total_size,cudaMemcpyHostToDevice);
    cudaMemcpy(cd,c,total_size,cudaMemcpyHostToDevice);
    struct timeval start, end;
    gettimeofday( &start, NULL );
    addVec1<<<dim3(32,32),dim3(32,32)>>>(ad,bd,cd);
    cudaDeviceSynchronize();
    gettimeofday( &end, NULL );

    cudaMemcpy(c,cd,total_size,cudaMemcpyDeviceToHost);
    for(int i=0;i<10;i++){
        cout<<c[i]<<" "<<c[len-1-i]<<" ";
    }
    cout<<endl;
    free(a);
    free(b);
    free(c);
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time is " << timeuse << "us" <<endl;
}