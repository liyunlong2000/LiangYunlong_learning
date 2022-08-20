#include<iostream>
#include <stdlib.h> 
#include <time.h>
#include <sys/time.h>
using namespace std;

const int m=1024*3;
const int n=32;
__global__ void mul(float* ad,float*bd,float*cd){
    __shared__ float ads[n][n];
    __shared__ float bds[n][n];
    // int block_size=(blockDim.x*blockDim.y)*sizeof(float);
    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    // printf("%d ",ad[1]);
    int temp=(m/blockDim.x)+1;
    for(int l=0;l<temp;l++){
        int i=bx*blockDim.x+tx;
        int j=by*blockDim.y+ty;
        int l1=l*blockDim.y+ty;
        int l2=l*blockDim.x+tx;
       if(i<m&&j<m&&l1<m&&l2<m){
        ads[tx][ty]=ad[i*m+l1];
        bds[tx][ty]=bd[l2*m+j];
        __syncthreads();
        for(int k=0;k<blockDim.x;k++){
            cd[i*m+j]+=ads[threadIdx.x][k]*bds[k][threadIdx.y];
        }
        __syncthreads();
       }
    }
}
int main(){
    float *a,*b,*c,*ad,*bd,*cd;
    int total_size=m*m*sizeof(float);
    a = (float*)malloc(total_size);
    b = (float*)malloc(total_size);
    c = (float*)malloc(total_size);
    cudaMalloc((void**)&ad,total_size);
    cudaMalloc((void**)&bd,total_size);
    cudaMalloc((void**)&cd,total_size);
    srand((unsigned)time(NULL));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            a[i*m+j]=20.3;
            b[i*m+j]=5.7;
        }
        
    }
    struct timeval start, end;
    gettimeofday( &start, NULL );
    cudaMemcpy(ad,a,total_size,cudaMemcpyHostToDevice);
    cudaMemcpy(bd,b,total_size,cudaMemcpyHostToDevice);
    cudaMemcpy(cd,c,total_size,cudaMemcpyHostToDevice);
    dim3 blockSize(32,32);
    dim3 gridSize(m/32+1,m/32+1);
    mul<<<gridSize,blockSize>>>(ad,bd,cd);
    cudaMemcpy(c,cd,total_size,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday( &end, NULL );

    for(int i=m-10;i<m;i++){
        for(int j=m-10;j<m;j++){
            cout<<c[i*m+j]<<" ";
        }
        cout<<endl;
    }
        cout<<"-------"<<endl;
        for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            cout<<c[i*m+j]<<" ";
        }
        cout<<endl;
    }
    free(a);
    free(b);
    free(c);
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time is " << timeuse/1000 << "ms" <<endl;
    return 0;
}