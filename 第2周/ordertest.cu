#include<iostream>

using namespace std;

__global__ void test(int *ad,int *bd,int * cd){
    int tx=threadIdx.x,ty=threadIdx.y;
    printf("cuda中二维数组的列优先:\n");
    int num[2][3],temp=0;
    // printf("%d(%d:%d) ",ad[tx*3+ty],tx,ty);
    printf("%d(%d:%d) ",ad[ty*2+tx],tx,ty);
    int test[2][3]={1,2,3,4,5,6};

    if(ty*2+tx==5){
        printf("\n");
        for(int i=0;i<6;i++){
            printf("%d ",(test[0]+i));
        }
        printf("\n");
        for(int i=0;i<2;i++){
            for(int j=0;j<3;j++){
                printf("%d ",&(test[i][j]));
            }
        } 
        printf("\n");
        __shared__ int test2[2][3];
        for(int i=0;i<2;i++){
            for(int j=0;j<3;j++){
                printf("%d ",&(test2[i][j]));
            }
        } 

    }
    // printf("cuda中二维数组的列优先:\n");
    // int num[2][3],temp=0;
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<3;j++){
    //         num[i][j]=(temp++);
    //     }
    // }
    // temp=0;
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<3;j++){
    //         printf("%d ",*(num[0]+(temp++)));
    //     }
    //     printf("\n");
    // }
    // printf("----------\n");
    // printf("复制到GPU中:\n");
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<3;j++){
    //         printf("%d ",ad[i*3+j]);
    //     }
    //     printf("\n");
    // }
    // printf("----------\n");
    // for(int i=0;i<3;i++){
    //     for(int j=0;j<2;j++){
    //         printf("%d ",bd[i*2+j]);
    //     }
    //     printf("\n");
    // }
    // printf("----------\n");
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<2;j++){
    //         int sum=0;
    //         for(int k=0;k<3;k++){
    //             sum+=ad[i*3+k]*bd[k*2+j];
    //         }
    //         cd[i*2+j]=sum;
    //     }
    // }
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<2;j++){
    //         printf("%d ",cd[i*2+j]);
    //     }
    //     printf("\n");
    // }
}
int main(){
    int a[2][3],temp=0,b[3][2],c[2][2];
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            a[i][j]=(temp);
            temp++;
        }
    }
    temp=0;
    for(int i=0;i<3;i++){
        for(int j=0;j<2;j++){
            b[i][j]=(temp);
            temp++;
        }
    }

    // cout<<"未复制到GPU中:"<<endl;
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<3;j++){
    //         printf("%d ",a[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("----------\n");
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<2;j++){
    //         int sum=0;
    //         for(int k=0;k<3;k++){
    //             sum+=a[i][k]*b[k][j];
    //         }
    //         c[i][j]=sum;
    //     }
    // }
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<2;j++){
    //         cout<<c[i][j]<<" ";
    //     }
    //     cout<<endl;
    // }

    int* ad,*bd,*cd;
    cudaMalloc((void**)&ad,6*sizeof(int));
    cudaMalloc((void**)&bd,6*sizeof(int));
    cudaMalloc((void**)&cd,4*sizeof(int));
    cudaMemcpy(ad,a,6*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(bd,b,6*sizeof(int),cudaMemcpyHostToDevice);
    test<<<dim3(1),dim3(2,3)>>>(ad,bd,cd);
    cudaMemcpy(c,cd,4*sizeof(int),cudaMemcpyDeviceToHost);
    // cout<<"从device中复制回来:"<<endl;
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<2;j++){
    //         cout<<c[i][j]<<" ";
    //     }
    //     cout<<endl;
    // }
}