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
对于连续的三个thread(0,0)、(1,0)和(0,1)，它们所访问的内存为(ad+0)、(ad+1)和(ad+2),访问内存顺序是规则的。但是(tx,ty)元素不是矩阵的(tx,ty)元。

为了顺序访问内存，可以将(ty,tx)看作矩阵的索引，只需要调换blockSize、gridSize中参数。例如对于一个二维矩阵{{0,1,2}{3,4,5}}和一个blockSize为(3,2)的block而言，
对于索引为(ty,tx)的thread，要想访问矩阵中的(ty,tx)元，应该按行优先顺序转为一维的序号，就能找到相应的元素。
```
ad[ty*2+tx]
```
上述只是以另一种坐标系看待矩阵。具体代码参考[ordertest.cu](ordertest.cu).
# 合并访存
合并访存可理解为：当warp中线程并行执行时，从global memory中读取的数据按块传输到SM中。这样当warp访问的数据在块中时，就不需要
额外进行数据传输，从而减少访存次数。
## 具体例子
举个例子(假设一次数据传输指的是将32字节的数据从全局内存通过32字节的缓存传输到SM，
且已知从全局内存转移到缓存的首地址一定是一个最小粒度(此处为32字节)的整数倍
(比如0 ~ 31、32 ~ 63、64 ~ 95这样传)，cudaMalloc分配的内存的首地址至少是256字节的整数倍)，
下面这两个函数，add1是合并访问的，观察其第一次传输，
第一个线程块中的线程束将访问x中的第0 ~ 31个元素，总共128字节的数据大小，
这样4次传输就可以完成数据搬运，而128/32=4，说明合并度为100%。
而add2则是非合并访问的，观察第一次传输，
第一个线程块中的线程束将访问x中的第1~32个元素，若x的首地址为256字节，
则线程束将作5次传输：256 ~ 287、288 ~ 319、320 ~ 351、352 ~ 383、384 ~ 415，其合并度为4/5=80%。
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
第二种方法为：参考[cuda列优先](https://github.com/liyunlong2000/LiangYunlong_learning/tree/main/%E7%AC%AC%E4%BA%8C%E5%91%A8#cuda%E4%B8%AD%E5%88%97%E4%BC%98%E5%85%88),以另一种索引方式考虑加法，对于全局索引为(tidy,tidx)的thread,其负责将ad(tidy,tidx)与bd(tidy,tidx)相加的结果保存到cd(tidy,tidx)中。
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
# shared memory和bank conflict
下面以矩阵转置为例，讨论合并访存、shared memory和bank conflict对计算效率的影响，学习优化技巧。
## 矩阵转置
对于m * n的矩阵，blockSize为(32,32)，考虑不同核函数的耗时。
### 写入合并版本transpose1
```
__global__ void transpose1(float*ad,float*cd){
    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int nx=tx+(bx*blockDim.x),ny=ty+(by*blockDim.y);
    if(nx<m&&ny<n){
        cd[nx+ny*m]=ad[nx*n+ny];
    }
}
```
对于tranpose1，nx是顺序递增的，因此访问global memory中ad是非合并访存，访问global memory中cd是合并访存。
### 读取合并版本transpose2
```
__global__ void transpose2(float*ad,float*cd){
    int tx=threadIdx.x,ty=threadIdx.y;
    int bx=blockIdx.x,by=blockIdx.y;
    int nx=tx+(bx*blockDim.x),ny=ty+(by*blockDim.y);
    if(nx<n&&ny<m){
        cd[nx*m+ny]=ad[nx+ny*n];
    }
}
``` 
对于tranpose2，nx是顺序递增的，因此访问global memory中ad是合并访存，访问global memory中cd是非合并访存。
### 使用shared memory优化读写合并版本transpose3
```
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
```
对于tranpose3,sm[bn][bn]的大小与blockSize(32,32)相同。nx是顺序递增的，因此访问global memory中ad是合并访存，访问global memory中cd也是合并访存。
### 优化shared memory中bank conflict版本transpose4
```
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
```
对于tranpose3,sm的大小为[bn][bn+1]。nx是顺序递增的，因此访问global memory中ad是合并访存，访问global memory中cd也是合并访存，但是消除了bank conflict。

**TODO:说明bank conflict和为什么解决了conflict**

### 实验结果
![image](https://user-images.githubusercontent.com/56336922/186067941-757a9208-1d13-4c4c-b832-339332f3ab11.png)

上表中单位为ms，具体数据参考[矩阵转置时间表.xlsx](矩阵转置时间表.xlsx)
![image](https://user-images.githubusercontent.com/56336922/186067907-5d0b4c6a-3e83-45b7-be98-b173c2a00bdd.png)

上图中横坐标为矩阵的大小，纵坐标为核函数耗时(ms)，可以得出以下几个结论：
1. transpose1和transpose2都是一次合并访存和一次非合并访存，但是两者效率差距明显。可能是GPU架构对读取数据做了相关优化，但未
对写入数据进行优化。
2. transpose1和transpose3的计算效率相当。使用shared memory虽然能提高访存效率，但是bank conflict和shared memory的写入、读取
会消耗额外的时间，使得两者之间计算效率相差不大。
3. transpose4较transpose3计算效率提高显著。通过解决bank conflict之后，计算时间显著减少，这说明bank conflict是shared memory中
影响计算效率的主要因素。

具体代码参考[matrixTranspose.cu](matrixTranspose.cu)
# 矩阵乘法访存优化
参考[第一周中矩阵乘法示例](https://github.com/liyunlong2000/LiangYunlong_learning/tree/main/%E7%AC%AC%E4%B8%80%E5%91%A8#cuda%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E4%BE%8B%E5%AD%90)，增加对matrixMul2中写入cd进行优化:
```
__global__ void mul1(float* ad,float*bd,float*cd){

        int i=threadIdx.x+(blockDim.x*blockIdx.x);
        int j=threadIdx.y+(blockDim.y*blockIdx.y);
        if(i<m && j<m){
        float sum=0;

        for(int k=0;k<m;k++){
            sum+=ad[j*m+k]*bd[k*m+i];
        }
        cd[j*m+i]=sum;
        }
    }
```
## 实验结果
![image](https://user-images.githubusercontent.com/56336922/186126086-c3b5c274-6ac7-4e8b-8e7b-f23d1aead99f.png)

matrixMul4中数据是对matrixMul2中访存进行优化后的结果，单位为ms。从结果可以看出，当cd的访存方式为合并访存时，计算效率提升显著。
# 参考资料
[CUDA学习(二)矩阵转置及优化(合并访问、共享内存、bank conflict)](https://zhuanlan.zhihu.com/p/450242129)

[NVIDIA CUDA初级教程视频](https://www.bilibili.com/video/BV1kx411m7Fk?p=12&vd_source=d759cf8f50c820c1f20e1c9049769dbc)
# docker入门
docker是一个应用容器引擎，我们可以将程序、运行环境和依赖文件打包到容器中，这样方便我们将程序移植到其它平台中，而不需要自己配置程序所需的运行环境。

## docker中基本概念
- 镜像(images)：包含了运行环境的只读模板。是一个最小的root文件系统，为程序提供所需的运行环境。
- 容器(container)：根据镜像创建的实例，可看做运行的Linux系统.
- 仓库(repository)：包含各种镜像的仓库，我们能够将镜像上传到仓库中或从仓库中拉取所需的镜像。
## docker中基本操作
![未命名文件 (5)](https://user-images.githubusercontent.com/56336922/186643375-76459282-1652-40c1-b817-ef89f3ff728c.png)

上图展示了docker中基本操作命令。
### 获取镜像
我们可以使用docker pull命令从仓库中拉取一个镜像。例如：
```
docker pull ubuntu:13.10
```
- ubuntu：仓库源的名字
- 13.10：':'后跟的标签表示其版本

### 推送镜像
使用docker push将镜像保存到仓库中。例如：
```
docker push runoob/ubuntu:18.04
```
runoob/ubuntu:18.04: 表示'用户名/仓库名:标签'

### 启动容器
使用docker run根据镜像创建容器。另外，可使用docker images查看本地中镜像。例如：
```
docker run -it ubuntu /bin/bash
```
- -it:表示-i和-t，两者一起使用能够以交互的形式操作容器的终端。
- ubuntu：镜像名
- /bin/bash： 以命令行的形式进入容器。

- -d:容器在后台运行。
### 更新镜像
镜像是一个只读的模板，对由镜像创建的容器进行修改并不会修改镜像。为了更新镜像，我们可以对改动后的容器使用docker comit命令保存为新的镜像。例如：
```
docker commit -m="has update" -a="runoob" e218edb10161 runoob/ubuntu:v2
```
- -m：表示镜像的描述信息
- -a：镜像的作者
- e218edb10161：容器的ID
- runoob/ubuntu:v2：镜像的名称和版本号

### 构建镜像
我们可以通过dockerfile文件和docker build命令创建一个镜像。例如：
```
FROM    centos:6.7
MAINTAINER      Fisher "fisher@sudops.com"

RUN     /bin/echo 'root:123456' |chpasswd
RUN     useradd runoob
RUN     /bin/echo 'runoob:123456' |chpasswd
RUN     /bin/echo -e "LANG=\"en_US.UTF-8\"" >/etc/default/local
EXPOSE  22
EXPOSE  80
CMD     /usr/sbin/sshd -D
```
上面是一个dockerfile。
- FROM：后面表示基础的镜像源
- MAINTAINER：表示镜像作者和信息
- RUN：构建镜像时执行的命令，会为镜像创建新的图层
- EXPOSE：容器暴露的端口，能够访问容器内的应用服务
- CMD：容器启动后执行的命令
根据dockerfile文件使用docker build命令创建一个镜像。例如：
```
docker build -t runoob/centos:6.7 .
```
- -t：表示镜像的名称和版本为runoob/centos:6.7
- .：表示dockerfile所在的目录为当前目录
### 导出容器
使用docker export命令将容器导出为.tar文件。例如：
```
docker export 1e560fca3906 > ubuntu.tar
```
上面命令将ID为1e560fca3906的容器导出为容器快照文件ubuntu.tar
### 导入容器快照
使用docker import将容器快照文件导入为镜像。例如：
```
docker import /tmp/ubuntu.tar test/ubuntu:v1
```
- /tmp/ubuntu.tar:文件目录
- test/ubuntu:v1:镜像名称和标签
## 参考资料
[菜鸟教程(docker)](https://www.runoob.com/docker/docker-tutorial.html)

[Docker入门](https://zhuanlan.zhihu.com/p/23599229)
 
 
# Swin Transfomer
![image](https://user-images.githubusercontent.com/56336922/187054770-86b86d68-5200-4343-b6bf-87de2471fb11.png)
上图左边为Swin Transfomer的架构图，右边为Swin Transfomer Block的结构图。

## Swin Transfomer推理过程
对于Swin Tiny的超参数C为96和block数量为[2,2,6,2]。Patch Size默认为4* 4，Windows size默认为7* 7。
### Patch Partition & Linear Embedding
```
输入图片的尺寸为(B,3,224,224)
输出张量大小为(B,96,56,56)
```
- Patch Partition：将输入张量按patch大小进行划分，并在第二个维度‘3’上拼接。因此输出为(B,3* 4* 4,224/4,224/4),后两维表示patch的数量，第二维‘48’表示patch的维度。

- Linear Embedding：对patch的维度‘48’映射为C=‘96’。因此输出为(B,96,56,56)

### Stage1 Swin Transformer Block
```
输入张量大小为(B,96,56,56)
输出张量大小为(B,96,56,56)
```
两个连续的Swin Transformer Block为一组，以实现window之间的通信。

- Block1：将输入张量的后两维合并，张量变为(B,96,56* 56).首先对张量进行window划分，每个window大小为7* 7，并在第一维’B’上拼接，变为(8* 8* B,96,56/ 7* 56/ 7),即(64B,96,49).然后对每个window进行MSA操作，先将(64B,96,49)复制三份(64B,96* 3,49),q、k、v的维度为(64B,96,49),根据多头划分为(64B,3,32,49)，最后执行完attention操作变为(64B,96,7,7)。然后通过window reverse变回(B,96,56,56)
- Block2：与Block1类似，只是SW-MSA中进行窗口移动和掩码操作，维度最终保持不变。

### Swin Transformer Block&Patch Merging
上述两个操作合为Swin Transfomer Basic Layer。后面stage2、stage3和stage4中Swin Transformer Block的操作和stage1中类似，不同的是Linear Embedding变为Patch Merging，以stage2中操作为例。
```
输入张量大小为(B,96,56,56)
输出张量大小为(B,192,28,28)
```
- Patch Merging：首先每隔一行一列取一个patch，合并为一个大的patch，并在第二维‘96’上拼接,张量变为(B,4* 96,56/2,56/2)，即(B,384,28,28)。然后通过1* 1的卷积将第二维‘384’变为192，即(B,192,28,28).
- Swin Transformer Block：类似stage1中操作，输入输出的张量保持不变。
因此，stage3中输入张量大小为(B,192,28,28)，输出张量大小为(B,384,14,14)。Stage4中输入张量大小为(B,384,14,14)，输出张量大小为(B,768,7,7)
### 参考资料
[注意力,多头注意力,自注意力](https://zhuanlan.zhihu.com/p/366592542#:~:text=%E7%9B%B8%E6%AF%94%E5%8D%95%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B%EF%BC%8C%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%8F%AF%E4%BB%A5%E8%80%83%E8%99%91%E5%88%B0%E6%9B%B4%E5%A4%9A%E5%B1%82%E9%9D%A2%E7%9A%84%E4%BF%A1%E6%81%AF%E3%80%82,%E5%BD%93%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%9A%84query%E5%92%8Ckey%E3%80%81value%E5%85%A8%E9%83%A8%E6%9D%A5%E8%87%AA%E4%BA%8E%E5%90%8C%E4%B8%80%E5%A0%86%E4%B8%9C%E8%A5%BF%E6%97%B6%EF%BC%8C%E5%B0%B1%E7%A7%B0%E4%B8%BA%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E3%80%82%20%E5%A6%82%E4%B8%8B%E5%9B%BE%E6%89%80%E7%A4%BA%EF%BC%8Cquery%E5%92%8Ckey%E3%80%81value%E5%85%A8%E9%83%BD%E6%9D%A5%E6%BA%90%E4%BA%8EX%E3%80%82)

[Swin Transformer详解](https://blog.csdn.net/weixin_41978699/article/details/121572267?spm=1001.2014.3001.5506)

[多头注意力及注意力](https://www.bilibili.com/video/BV15v411W78M/?spm_id_from=333.788&vd_source=d759cf8f50c820c1f20e1c9049769dbc)

