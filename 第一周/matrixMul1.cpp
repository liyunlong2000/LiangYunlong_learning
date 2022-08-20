#include<iostream>
#include <stdlib.h> 
#include <time.h>
#include <sys/time.h>
using namespace std;

const int m=3042;
int main(){
    float *a,*b,*c;
    int total_size=m*m*sizeof(float);
    a = (float*)malloc(total_size);
    b = (float*)malloc(total_size);
    c = (float*)malloc(total_size);
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
    for(int i=0;i<m;i++){
        for(int j=0;j<m;j++){
            float sum=0;
            for(int k=0;k<m;k++){
                sum+=a[i*m+j]*b[i*m+j];
            }
            c[i*m+j]=sum;
        }
    }
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
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time is " << timeuse/1000 << "ms" <<endl;

    return 0;
}