#include<iostream>
#include <stdlib.h> 
#include <time.h>
#include <sys/time.h>
using namespace std;

const int m=1024*30;
const int n=1024*10;
const int bn=32;

//写合并
__global__ void transpose1(float*ad,float*cd){
    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int nx=tx+(bx*blockDim.x),ny=ty+(by*blockDim.y);
    if(nx<m&&ny<n){
        cd[nx+ny*m]=ad[nx*n+ny];
    }
}
//读合并
__global__ void transpose2(float*ad,float*cd){
    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int nx=tx+(bx*blockDim.x),ny=ty+(by*blockDim.y);
    if(nx<n&&ny<m){
        cd[nx*m+ny]=ad[nx+ny*n];
    }
}
//使用shared memory优化读写合并
__global__ void transpose3(float*ad,float*cd){
    __shared__ float sm[bn][bn];
    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int nx=tx+(bx*blockDim.x),ny=ty+(by*blockDim.y);
    if(nx<n&&ny<m){
        sm[ty][tx]=ad[nx+ny*n];
        __syncthreads();
    }
    int nx1=tx+(by*blockDim.y),ny1=(bx*blockDim.x)+ty;
    if(ny1<n&&nx1<m){
        cd[ny1*m+nx1]=sm[tx][ty];
    }
    
}
//优化shared memory中bank conflict
__global__ void transpose4(float*ad,float*cd){
    __shared__ float sm[bn][bn+1];
    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int nx=tx+(bx*blockDim.x),ny=ty+(by*blockDim.y);
    if(nx<n&&ny<m){
        sm[ty][tx]=ad[nx+ny*n];
        __syncthreads();
    }
    int nx1=tx+(by*blockDim.y),ny1=(bx*blockDim.x)+ty;
    if(ny1<n&&nx1<m){
        cd[ny1*m+nx1]=sm[tx][ty];
    }
    
}
int main(){
    float *a,*b,*c,*ad,*bd,*cd;
    int total_size=m*n*sizeof(float);
    a = (float*)malloc(total_size);
    c = (float*)malloc(total_size);
    cudaMalloc((void**)&ad,total_size);
    cudaMalloc((void**)&cd,total_size);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a[i*n+j]=i;
        }
        
    }

    // for(int i=0;i<64;i++){
    //     for(int j=0;j<64;j++){
    //         cout<<a[i*n+j]<<" ";
    //     }
    //     cout<<endl;
    // }

    struct timeval start, end;
    cudaMemcpy(ad,a,total_size,cudaMemcpyHostToDevice);
    cudaMemcpy(cd,c,total_size,cudaMemcpyHostToDevice);
    
    gettimeofday( &start, NULL );
    transpose1<<<dim3(m/32+1,n/32+1),dim3(32,32)>>>(ad,cd);
    cudaDeviceSynchronize();
    gettimeofday( &end, NULL );
    cudaMemcpy(c,cd,total_size,cudaMemcpyDeviceToHost);
    // cout<<"-------"<<endl;
    // for(int i=0;i<100;i++){
    //     for(int j=0;j<100;j++){
    //         cout<<c[i*m+j]<<" ";
    //     }
    //     cout<<endl;
    // }
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "transpose1 total time is " << timeuse/1000 << "ms" <<endl;

    gettimeofday( &start, NULL );
    transpose2<<<dim3(n/32+1,m/32+1),dim3(32,32)>>>(ad,cd);
    cudaDeviceSynchronize();
    gettimeofday( &end, NULL );
    cudaMemcpy(c,cd,total_size,cudaMemcpyDeviceToHost);


    timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "transpose2 total time is " << timeuse/1000 << "ms" <<endl;
    // for(int i=0;i<100;i++){
    //     for(int j=0;j<100;j++){
    //         cout<<c[i*m+j]<<" ";
    //     }
    //     cout<<endl;
    // }
    gettimeofday( &start, NULL );
    transpose3<<<dim3(n/32+1,m/32+1),dim3(32,32)>>>(ad,cd);
    cudaDeviceSynchronize();
    gettimeofday( &end, NULL );
    cudaMemcpy(c,cd,total_size,cudaMemcpyDeviceToHost);
    // for(int i=0;i<100;i++){
    //     for(int j=0;j<100;j++){
    //         cout<<c[i*m+j]<<" ";
    //     }
    //     cout<<endl;
    // }
    timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "transpose3 total time is " << timeuse/1000 << "ms" <<endl;

    gettimeofday( &start, NULL );
    transpose4<<<dim3(n/32+1,m/32+1),dim3(32,32)>>>(ad,cd);
    cudaDeviceSynchronize();
    gettimeofday( &end, NULL );
    cudaMemcpy(c,cd,total_size,cudaMemcpyDeviceToHost);
    // for(int i=0;i<100;i++){
    //     for(int j=0;j<100;j++){
    //         cout<<c[i*m+j]<<" ";
    //     }
    //     cout<<endl;
    // }
    timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "transpose4 total time is " << timeuse/1000 << "ms" <<endl;
    cout<<"-------"<<endl;
    free(a);
    free(c);
    cudaFree(ad);
    cudaFree(cd);
    return 0;
}