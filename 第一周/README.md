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
上图为GTX 480中一个核心，每个黄色方块代表一个计算单元，类似于CPU中ALU。十二个计算单元为一组，每一组共享一个指令流，由橙色方块进行取指、译码。执行上下文中包含若干个需要完成的任务。每个SM含有一块共享内存，由该SM中所有计算单元所共享。
## GPU的存储器设计
![QQ截图20220818104329](https://user-images.githubusercontent.com/56336922/185281405-5b2ce319-9e73-4474-918e-c98de0948790.png)

上图为GPU存储器的层次结构图，每个SM中包含了一个纹理缓存、一个共享内存(L1缓存)。SM外为一个L2缓存和一个显存，L2缓存和显存之间可以数据交互。

![QQ截图20220818105050](https://user-images.githubusercontent.com/56336922/185282317-457b7655-8d24-4ae7-abd3-0144886aca9a.png)

上图为GPU存储器硬件上层次结构图，其中WorkLtem相当于一个计算单元，Compute Unit相当于一个SM。这样绿色方块相当于共享内存，橘色方块相当于L2缓存，红色方块为显存。
## CPU和GPU交互模式
![QQ截图20220818105643](https://user-images.githubusercontent.com/56336922/185283088-203b3c46-b97a-418f-b038-96fabfdabafd.png)
如上图所示，CPU和GPU有各自物理内存空间，通过PCIE总线互连。
## GPU线程组织模型
![QQ截图20220818110103](https://user-images.githubusercontent.com/56336922/185283578-e24b58ed-a197-4493-9cec-1a6563ec072b.png)
上图展示了GPU的线程组织模型。一个Kernel意为GPU上执行的函数，具有大量的线程。Kernel启动一个grid，包含若干个block，每个block包含若干个线程。一个block内部的线程共享share memory。

![QQ截图20220818110619](https://user-images.githubusercontent.com/56336922/185284171-f23125d8-bfc5-4c85-b712-acc6773ae4b9.png)
上图展示了GPU与线程的映射关系。一个线程对应GPU中一个线程处理器(计算单元)，一个block对应GPU中一个Multi-processor(SM),一个gird对应了显示设备(gpu)
## GPU存储模型
![QQ截图20220818111131](https://user-images.githubusercontent.com/56336922/185284814-3b079c74-3acc-43ab-b077-70f8243ddd2a.png)

上图展示了GPU内存和线程的对等关系。每个线程对应私有的寄存器、local memory，每个block对应共享内存，每个grid对应设备的global memory

![QQ截图20220818111431](https://user-images.githubusercontent.com/56336922/185285124-67acb06d-cee7-40a1-be5c-6c7403753242.png)
## 参考视频
[NVIDIA CUDA初级教程视频](https://www.bilibili.com/video/BV1kx411m7Fk?p=5&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=d759cf8f50c820c1f20e1c9049769dbc)
