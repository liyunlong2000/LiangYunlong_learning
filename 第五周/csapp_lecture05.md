# 机器级别编程 2：控制
## 条件代码
### 处理器状态
![image](https://user-images.githubusercontent.com/56336922/189527633-f4c89802-c618-4b76-93b8-f2c0779a64b6.png)

上图展示了x64架构下的寄存器，存放着当前执行程序的信息。
- 临时数据：存储在%rax，...
- 运行时栈的位置：存储在%rsp中
- 当前代码的控制点：存储在%rip(x64的pc)，...
- 最近运算的状态：存储在CF、ZF、SF、OF中
### 条件代码
CF、ZF、SF、OF都是单位的寄存器，存储关于最近运算的状态，用于实现控制流跳转。
- CF(Carry flag)：针对无符号数的进位标识符。
  - 隐式设置：如果两个无符号数相加，在最高位还需要进位（也就是溢出了），那么 CF 标识位就会被设置
  - 显式设置(cmpq b,a -> a-b)：如果在最高位还需要进位（也就是溢出了），那么 CF 标识位就会被设置
- SF(Sign	Flag)：符号数的符号标志符。
  - 隐式设置：最近运算结果等于 0，那么 ZF 标识位会被设置
  - 显式设置(cmpq b,a -> a-b)：如果运算结果小于0
  - 显式设置(testq b,a -> a&b)：如果a&b<0
- ZF(Zero flag):零标识符
  - 隐式设置： 最近运算结果小于 0，那么 SF 标识位会被设置
  - 显式设置(cmpq b,a -> a-b)：如果a==b，那么 SF 标识位会被设置
  - 显式设置(testq b,a -> a&b)：如果a&b==0
- OF(Overflow flag):符号数的溢出标识符。
  - 隐式设置：二进制补码运算结果溢出，那么 OF 标识位会被设置
  - 显式设置(cmpq b,a -> a-b)：运算结果溢出，那么 OF 标识位会被设置

![image](https://user-images.githubusercontent.com/56336922/189528579-095a1e5f-ca24-402f-bfe2-64aadad38ef2.png)

SetX系列指令通过以上各种条件码的组合，实现一些逻辑判断。
- 它只改变目标操作数的低1位字节，不改变高7位字节
- 我们可以引用寄存器的最低位字节

### 条件代码例子
![image](https://user-images.githubusercontent.com/56336922/189528740-8f2933be-f706-4462-953b-4a74dea75d4b.png)

- 首先通过cmpq指令，显示设置条件码
- 然后通过setq指令,根据条件码进行逻辑判断，设置目标操作数。
- 将目标操作数移动到%rax中低32位%eax中，并返回
  - movzbl是32位指令，会设置高32位为0
## 件分支
### 跳转
![image](https://user-images.githubusercontent.com/56336922/189529317-f89c7446-5b1d-41cf-b83a-e408a6be4a48.png)

jX系列通过以上各种条件码的组合，实现控制跳转。
