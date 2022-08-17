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

## GPU架构
### 为什么需要GPU?
### 三种方法提升GPU的处理速度
### 实际GPU设计举例
### GPU的存储器设计

## GPU编程模型
### CPU和GPU互动模式
### GPU线程组织模型
### GPU存储模型
