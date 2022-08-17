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
使用cublas中gemm进行矩阵相乘，可以理解为以下过程：1.矩阵根据传入参数，进行列优先转换。2.矩阵相乘。3.矩阵进行行优先转换
### 
