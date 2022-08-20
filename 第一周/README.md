# gemm中参数理解
c/c++、python中高维数组以行优先顺序存储，cuda中高维数组以列优先顺序存储.对于矩阵乘法而言，直接将两个矩阵传入cuda中实现矩阵乘法，会使矩阵中元素位置发生改变。cublas中gemm提供若干个参数对矩阵进行调整。以矩阵A * B=C为例
## M、N和K
M:矩阵A的行数

K:矩阵A和列数、矩阵B的行数

N:矩阵B的列数
## lda、ldb和ldc
ld意为leading dimension，lda、ldb和ldc分别代表矩阵A、B和C的主维
## trans_a和trans_b
trans_a:是否对矩阵A进行转置

trans_b:是否对矩阵B进行转置
## 运算过程
使用cublas中gemm进行矩阵相乘，可以理解为以下过程：
1. 矩阵根据传入参数，进行列优先转换。
2. 矩阵相乘。
3. 矩阵进行行优先转换
### 列优先转换
将行优先存储变为列优先存储，并且根据参数改变矩阵的形状：
- 若trans_a为N，矩阵A的形状为M * K
- 若trans_a为T，矩阵A的形状为lda * M
- 若trans_b为N，矩阵B的形状为K * N
- 若trans_b为T，矩阵B的形状为ldb * K
### 矩阵相乘
根据trans参数，对相应矩阵进行转置，然后进行矩阵乘法运算。
### 行优先转换
矩阵相乘后的结果C由列优先存储转为行优先存储，并根据参数改变矩阵的形状：
- 根据参数ldc，调整C的列数为ldc，行数为C的原列数
## 具体列子
![image](https://user-images.githubusercontent.com/56336922/185171103-6b9c6187-9dde-4963-a76b-e43c7742b8ce.png)

## 参考链接
[有关CUBLAS中的矩阵乘法函数](https://www.cnblogs.com/cuancuancuanhao/p/7763256.html)

# CUDA硬件基础
主要介绍CUDA编程中所需的GPU相关知识。
## 为什么需要GPU?
CPU的设计以逻辑处理和计算为主，核心的设计针对逻辑进行了特定优化，适用于串行处理数。为了解决图像、气象等领域中的计算问题，产生了GPU。其适合大规模并行处理数据。
## 实际GPU设计举例
![QQ截图20220818103209](https://user-images.githubusercontent.com/56336922/185279920-a72b82cc-74a8-4e96-b1b4-9f4dd8ff03d0.png)
上图为GTX 480中一个核心，每个黄色方块代表一个处理单元，类似于CPU中ALU。十二个处理单元为一组，每一组共享一个指令流，由橙色方块进行取指、译码。执行上下文中包含若干个需要完成的任务。每个SM含有一块共享内存，由该SM中所有处理单元所共享。
## GPU的存储器设计
![QQ截图20220818104329](https://user-images.githubusercontent.com/56336922/185281405-5b2ce319-9e73-4474-918e-c98de0948790.png)

上图为GPU存储器的层次结构图，每个SM中包含了一个纹理缓存、一个共享内存(L1缓存)。SM外为一个L2缓存和一个显存，L2缓存和显存之间可以数据交互。

![QQ截图20220818105050](https://user-images.githubusercontent.com/56336922/185282317-457b7655-8d24-4ae7-abd3-0144886aca9a.png)

上图为GPU存储器硬件上层次结构图，其中WorkLtem相当于一个计算单元，Compute Unit相当于一个SM。这样绿色方块相当于共享内存，橘色方块相当于L2缓存，红色方块为显存。
## 参考资料
[NVIDIA CUDA初级教程视频](https://www.bilibili.com/video/BV1kx411m7Fk?p=5&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=d759cf8f50c820c1f20e1c9049769dbc)

# cude基础
## CPU和GPU交互模式
![QQ截图20220818105643](https://user-images.githubusercontent.com/56336922/185283088-203b3c46-b97a-418f-b038-96fabfdabafd.png)
如上图所示，CPU和GPU有各自物理内存空间，通过PCIE总线互连。
## GPU线程组织模型
![QQ截图20220818110103](https://user-images.githubusercontent.com/56336922/185283578-e24b58ed-a197-4493-9cec-1a6563ec072b.png)
上图展示了GPU的线程组织模型。一个Kernel意为GPU上执行的函数，具有大量的线程。Kernel启动一个grid，包含若干个block，每个block包含若干个线程。一个block内部的线程共享share memory。

- gridDim：dim3类型，表示一个grid中block块的个数
- blockDim：dim3类型，表示block的大小
- blockIdx：uint3来下，表示线程所在block的索引
- threadIdx：uint3类型，表示线程在block中的索引

线程的全局索引为：块地址* 块大小+块内线程地址。块地址由blockIdx和gridDim表示，块大小由blockDim表示，块内线程地址由threadIdx和blockDim表示。


![QQ截图20220818110619](https://user-images.githubusercontent.com/56336922/185284171-f23125d8-bfc5-4c85-b712-acc6773ae4b9.png)
上图展示了GPU与线程的映射关系。一个线程对应GPU中一个线程处理器(计算单元)，一个block对应GPU中一个Multi-processor(SM),一个gird对应了显示设备(gpu)
## GPU存储模型
![QQ截图20220818111131](https://user-images.githubusercontent.com/56336922/185284814-3b079c74-3acc-43ab-b077-70f8243ddd2a.png)

上图展示了GPU内存和线程的对等关系。每个线程对应私有的寄存器、local memory(作用域是每个thread，但实际上是global memry中存储空间)，每个block对应共享内存，每个grid对应设备的global memory

![QQ截图20220818111431](https://user-images.githubusercontent.com/56336922/185285124-67acb06d-cee7-40a1-be5c-6c7403753242.png)

## cuda编程框架
1. 使用cudaMalloc在device中为输入和输出分配一块内存，然后使用cudaMemcpy将输入从host拷贝到device中
2. 核函数执行，结果输出到所分配的内存中
3. 使用cudeMemcpy将输出从device拷贝到host中，然后使用cudaFree释放device中分配的内存

## cuda编程
### 函数声明
- __global__ ：表示只能从host端调用，在device端执行
- __device__ : 表示只能从device端调用，在device端执行
- __host__ : 表示只能从host端调用，在host端执行

### 向量数据类型
- char[1-4],uchar[1-4]
- int[1-4],uint[1-4]
- long[1-4],ulong[1-4]
- float[1-4]
- double1,double2

向量数据类型适用于host和device代码，通过函数make_<type name>构造，例如：
- int2 i2=make_int2(1,2);
- float4 f4=make_float4(1.0f,2.0f,3.0f,4.0f);
  
通过.x,.y,.z和.w来访问，例如：
- int x=i2.x
- int y=i2.y

### 线程同步
块内线程可以同步，使用_syncthreads创建一个barrier，块内所有线程将会在barrier处同步，以解决数据依赖的问题。块内同步是效率和开销的折中，有利于提高系统的可扩展性。
  
### 线程调度
软件上所启动的线程数往往大于SP的数量，即block中thread数大于SM中SP的数量，不能以block作为一个调度的基本单位，而是以一个更小的单位Warp。Warp是块内的一组线程。每个Warp运行于同一个SM上，是线程调度的基本单位。
  
线程调度的目的是延迟隐藏，提高计算效率。
  ![QQ截图20220818225631](https://user-images.githubusercontent.com/56336922/185432055-a4c1429d-1414-454e-92ab-ae5a185223a1.png)

同一个Warp中的线程执行步调是一致、同步的，对于分支结构，往往会导致部分线程处于停滞状态，计算效率下降。
  
对于GT80，每个Warp有32个线程，但每个SM只有8个SP。这样当一个SM调度一个Warp时，分为以下阶段：
1. 指令已经预备
2. 第一个周期8个线程进入SPs
3. 第二、三、四个周期各进入8个线程

因此，分发一个Warp需要四个周期。
### 内存模型
 寄存器是每个线程专用，若增加kernel的寄存器数量，可能会造成block数量减少而使thread数量减少。
  ![QQ截图20220818231043](https://user-images.githubusercontent.com/56336922/185432018-4c3fc13e-1ba7-4d93-89a5-c97b397a1c70.png)

  上图表示变量的声明。存储器表示变量所对应的存储器，作用域表示变量有效的作用范围，生命期表示变量内存分配和释放的的阶段。
## 参考资料
[NVIDIA CUDA初级教程视频](https://www.bilibili.com/video/BV1kx411m7Fk?p=5&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=d759cf8f50c820c1f20e1c9049769dbc)

[cuda中threadIdx、blockIdx、blockDim和gridDim的使用](https://www.cnblogs.com/tiandsp/p/9458734.html)
# cuda矩阵乘法例子
本节以矩阵相乘为例子，学习cuda编程。并且比较CPU版本和GPU版本下矩阵相乘的时间，体会GPU并行运算的加速效果。只考虑常数方阵的情况。
## CPU版本
对结果矩阵的(i,j)元素进行遍历，其值为"ik,kj->ij".详细代码见[matrixMul1.cpp](matrixMul1.cpp)
## GPU版本
### 未优化版本
GPU版本的写法参考[cuda编程框架](https://github.com/liyunlong2000/LiangYunlong_learning/edit/main/%E7%AC%AC%E4%B8%80%E5%91%A8/README.md#cuda%E7%BC%96%E7%A8%8B%E6%A1%86%E6%9E%B6),每个block块的维度为(32,32),每个grid的维度为(m/32+1,m/32+1),其中m为方阵的宽。核函数的思路为：对于每个thread,找到其在grid中的索引(i,j)，则该thread负责计算结果矩阵的(i,j)元，其值为"ik,kj->ij".需要注意代码中矩阵是用一维float数组表示，写法需要相应修改，详细代码见[matrixMul2.cu](matrixMul2.cu)
### 优化后版本
注意到在cuda中分配的内存存储在global memory中，对于结果矩阵的(i,j)元，每个thread从global memory中读取了一行(m个元素)，一列(m个元素),访存次数较多，计算次数占比小，计算效率较低。优化的思路为：每个block对应结果矩阵的一部分，对于block中每个thread，读入两个数据，这样将两个block大小的子矩阵存储到shared memory中。然后将子矩阵相乘的结果加到对应的block中。上述过程执行(m/blockDim.x+1)轮。

对于上述思路，每个thread从global memory中读取了m/block+m/block个元素，访存次数减少，并且访问shared memory的速度较快。需要注意使用_syncthreads()同步线程，从global memory中将数据拷贝到子矩阵中的思路为：先找到元素在grid中索引(i,j)，然后将索引转到一维形式。详细代码见[matrixMul3.cu](matrixMul3.cu)
### 实验结果
![image](https://user-images.githubusercontent.com/56336922/185744429-c75d3e9a-3df1-43f5-93f9-d8ef7fedc413.png)

上图展示了三种版本的矩阵乘法在不同维度下的计算效率，单位为ms。可以看到CPU版本下计算效率低下，使用GPU进行计算，效率提高了一百倍。并且优化后的版本计算效率提升明显，今后进行cuda编程需要考虑如何优化访存延迟。

