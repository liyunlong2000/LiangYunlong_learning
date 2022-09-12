# reduction
## 代码中reduction
![image](https://user-images.githubusercontent.com/56336922/189603817-a3ae4851-9618-4ca1-b4f4-16ce4232b491.png)

- 处理的维度布局为：**cu bjnu->cbjn;**

## 第一、二、三、四阶段reduction
![image](https://user-images.githubusercontent.com/56336922/189604177-c7cc840e-2165-4d3b-a22e-6f782cf8f4db.png)

- 处理的维度布局为：**uc ubjn->cbjn;**
- 代码调整为：
  - 对矩阵A进行转置，从cu->uc.
  - 对矩阵B进行转置，从bjnu->ubjn
  - 改变GEMM调用参数为：
     - trans_a:N   
     - trans_b:T      
     - M:2dim  
     - N:b * h * w
     - K:4dim   
     - lda:2dim
     - ldb:b * h * w   
     - ldc:2dim

# proj
## 代码中proj
![image](https://user-images.githubusercontent.com/56336922/189604581-d680d26a-81bd-48d8-ac07-9bcf87a5d5b2.png)

- 处理的维度布局为：**li bjni->lbjn;**

## 第一、二、三、四阶段proj
![image](https://user-images.githubusercontent.com/56336922/189604711-26ae457e-7c6e-4ffc-b6c0-b13a9e53789b.png)

- 处理的维度布局为：**il ibjn->lbjn;**
- 代码调整为：
  - 对矩阵A进行转置，从li->il.
  - 对矩阵B进行转置，从bjni->ibjn
  - 改变GEMM调用参数为：
     - trans_a:N   
     - trans_b:T      
     - M:dim  
     - N:b * h * w
     - K:dim   
     - lda:dim
     - ldb:b * h * w   
     - ldc:dim

# QKV-fused
## 代码中QKV-fused
![image](https://user-images.githubusercontent.com/56336922/189604964-05b0f25a-935e-4ba6-a7e3-efe4a342c82e.png)

- 处理的维度布局为：**iqph bjni->qphbjn;**

## 第一、三、四阶段QKV-fused
![image](https://user-images.githubusercontent.com/56336922/189605072-a824c02f-3d71-4d89-89bc-14c981068e11.png)

- 处理的维度布局为：**il ibjn->lbjn;**
- 代码调整为：
  - 对矩阵B进行转置，从bjni->ibjn
  - 改变GEMM调用参数为：
     - trans_a:N   
     - trans_b:T      
     - M:3dim  
     - N:b * h * w
     - K:dim   
     - lda:3dim
     - ldb:b * h * w   
     - ldc:3dim
## 第二阶段QKV-fused
![image](https://user-images.githubusercontent.com/56336922/189605365-5fb46cfa-5c08-4eac-b907-490101a1f536.png)

- 处理的维度布局为：**iqph bjni->qphbjn;**
- 代码调整为：
  - 与原代码中GEMM调用相同
