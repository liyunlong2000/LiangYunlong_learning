# cuda中列优先
cuda中列优先可以理解为thread的编号是按列优先的顺序编号的，而核函数中高维数组分配的内存仍是行优先顺序编号的。
cuda中基本执行单位为warp，每个warp由连续的32个thread组成。cuda中thread的编号是列优先的，即对于一个2D的block，其编号为：
```
int tid=threadIdx.y*blockDim.y+threadIdx.x;
```
对于一个二维矩阵{{0,1,2}{3,4,5}}和一个blockSize为(2,3)的block而言，对于索引为(tx,ty)的thread，要想访问矩阵中的(tx,ty)元，应该
按行优先顺序转为一维的序号，就能找到相应的元素。例如：
```
ad[tx*3+ty]
```
但是对于一个warp而言，上述访问内存的方式是离散的。即warp中连续的thread访问的内存空间不是连续的(非合并访存)，这样会降低
计算效率。因为thread的编号按tx递增，对于连续的三个thread(0,0)、(1,0)和(0,1)，它们所访问的内存为(ad+0)、(ad+3)和(ad+1),
访问顺序并不规则。

要想按矩阵分配的内存顺序访问矩阵，应将(tx,ty)转为按列优先顺序的一维序号。例如：
```
ad[ty*2+tx]
```
对于连续的三个thread(0,0)、(1,0)和(0,1)，它们所访问的内存为(ad+0)、(ad+1)和(ad+2)。但是(tx,ty)元素不是矩阵的(tx,ty)元。
具体代码参考[ordertest.cu](ordertest.cu]).
# 合并访存
合并访存可理解为：当warp中线程并行执行时，从global memory中读取的数据按块传输到SM中。这样当warp访问的数据在块中时，就不需要
额外进行数据传输。
## 具体例子
举个例子(假设一次数据传输指的是将32字节的数据从全局内存通过32字节的缓存传输到SM，
且已知从全局内存转移到缓存的首地址一定是一个最小粒度(此处为32字节)的整数倍
(比如0~31、32~63、64~95这样传)，cudaMalloc分配的内存的首地址至少是256字节的整数倍)，
下面这两个函数，add1是合并访问的，观察其第一次传输，
第一个线程块中的线程束将访问x中的第0~31个元素，总共128字节的数据大小，
这样4次传输就可以完成数据搬运，而128/32=4，说明合并度为100%。
而add2则是非合并访问的，观察第一次传输，
第一个线程块中的线程束将访问x中的第1~32个元素，若x的首地址为256字节，
则线程束将作5次传输：256~287、288~319、320~351、352~383、384~415，其合并度为4/5=80%。
```
//x, y, z为cudaMalloc分配全局内存的指针

void __global__ add1(float *x, float *y, float *z){
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    z[n] = x[n] + y[n];
}

void __global__ add2(float *x, float *y, float *z){
    int n = threadIdx.x+ blockIdx.x * blockDim.x + 1;
    z[n] = x[n] + y[n];
}

add1<<<128, 32>>>(x, y, z);
add2<<<128, 32>>>(x, y, z);
```
## 常数矩阵相加实验
对于两个1024 * 1024的常值方阵ad、bd，考虑使用两种方法进行相加。

第一种方法为:对于全局索引为(tidx,tidy)的thread,其负责将ad(tidx,tidy)与bd(tidx,tidy)相加的结果保存到cd(tidx,tidy)中。
```
__global__ void addVec(int *ad,int*bd,int*cd){
    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int tidx=tx+bx*blockDim.x,tidy=ty+by*blockDim.y;
    cd[tidx*1024+tidy]=ad[tidx*1024+tidy]+bd[tidx*1024+tidy];
}
```
第二种方法为：对于全局索引为(tidx,tidy)的thread,其负责将ad(tidy,tidx)与bd(tidy,tidx)相加的结果保存到cd(tidy,tidx)中。
```
__global__ void addVec1(int *ad,int*bd,int*cd){
    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int tidx=tx+bx*blockDim.x,tidy=ty+by*blockDim.y;
    cd[tidy*1024+tidx]=ad[tidy*1024+tidx]+bd[tidy*1024+tidx];
}
```
### 实验结果
```
(addVec)total time is 188us
(addVec1)total time is 120us
```
可以看出，按列顺序访问global memory的addVec1()方法计算效率高于按行顺序访问的addVec()方法。因为thread的编号是
按列顺序编号的，addVec1()方法访问的是连续的内存，能够合并访存，减少访存次数，提高计算效率。而addVec()方法
中访问的离散的内存，访存次数较多，计算效率低。

具体代码参考[memAcess.cu](memAcess.cu)