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
### 条件分支例子
![image](https://user-images.githubusercontent.com/56336922/189566567-192075f4-5c68-42dd-b0a0-47982882808a.png)

- 首先通过cmpq指令，显示设置条件码
- jle指令表示当x小于等于y时，跳转到标签.L4

### 条件移动
处理器支持条件移动指令，支持`if	(Test)	Dest <- Src`的操作。对于条件分支来说，先分别计算各分支中值，最后再根据条件取出分支中计算出来的值。
- 条件移动不会破坏指令流流水线，不需要控制流转移。因此，效率较高，GCC也通常使用它。但是，对于一些特殊的场合，条件移动并不适用。
    1.  `val = Test(x) ? Hard1(x) : Hard2(x); `两个分支的计算量很大
    2.  `val = p ? *p : 0; `可能会有副作用，当指针p为空时
    3.  `val = x > 0 ? x*=7 : x+=3;`同时修改了x的值，导致结果错误

### 条件移动例子
![image](https://user-images.githubusercontent.com/56336922/189567745-244961ec-b786-43b3-b836-2d0593218f9c.png)

 与条件分支相比，不需要控制转移，分别计算出两个分支的值，最后根据条件码，选取正确值。
 
 ## 循环
 ### Do-While循环例子
 ![image](https://user-images.githubusercontent.com/56336922/189568123-483c258c-9d2b-47be-ac51-46dbfd6e215d.png)

- jne指令根据指令`shrq %rdi`运算结果判断是否要继续循环。

### While循环例子
![image](https://user-images.githubusercontent.com/56336922/189568518-0159ff2b-137c-474d-99a3-6ccf5f0d48fb.png)

- 使用了Do-While循环的结构，在执行循环语句之前直接跳转到test块，由test块判断是否要继续循环。

![image](https://user-images.githubusercontent.com/56336922/189568692-858c5d35-895e-4f15-85b1-944968b7d8bd.png)

- 也可以在执行循环语句之前做一个判断，判断是否要执行循环语句或跳转到结束块。

### For循环例子
![image](https://user-images.githubusercontent.com/56336922/189568903-c60b4fe1-d0f4-4ef4-81a0-02586bef5d62.png)

for循环可以转换为While循环的形式。

![image](https://user-images.githubusercontent.com/56336922/189568996-820d6add-5eae-4cc2-82a4-5ad858260943.png)

一些特殊情况下，可以转换为Do-While循环，因为对于for循环来说，初始判断通常是成立的。这样While循环就可以转换为Do-While循环。

## Switch语句
### 跳转表结构
![image](https://user-images.githubusercontent.com/56336922/189569608-b47f04fe-d6c6-4a16-938c-ca2bf602423f.png)

 跳转表jtab包含了各代码块的起始地址Targ，可以根据代码块的索引找到相应代码块的起始地址。
### Switch 语句例子
![image](https://user-images.githubusercontent.com/56336922/189569963-6cfab74c-49d7-444c-aa2e-7ef8c47c420c.png)

上图的switch语句包含了多数的特殊情况：
- 共享条件：5和6
- 贯穿：2
- 缺失：4

![image](https://user-images.githubusercontent.com/56336922/189570276-f80f93da-27d8-4dd7-9595-60a2275bd3f0.png)

首先将switch中值x与case中最大值做比较。使用无符号的`ja`指令判断x与6的大小，并进行跳转。这样当x大于6或小于0时，都会跳转到
标签.L8。

```
.section .rodata
  .align 8 
.L4: 
  .quad .L8 # x = 0 
  .quad .L3 # x = 1 
  .quad .L5 # x = 2 
  .quad .L9 # x = 3 
  .quad .L8 # x = 4 
  .quad .L7 # x = 5 
  .quad .L7 # x = 6
```
函数的跳转表如上。
- 表中每个Targ占8字节
- .L4：为基地址

指令`jmp *.L4(,%rdi,8)`实现了间接跳转。
- 有效地址为：.L4 + x * 8
- * ：表示根据上面的有效地址进行寻址，实现了间接寻址。

![image](https://user-images.githubusercontent.com/56336922/189571310-f3fa4466-7428-43fa-906d-c5e519428f8f.png)

对`x==2，x==3`来说,`case 2`中并不需要w的初始值，将直接跳转到标签.L6中。而`case 3`中需要初始化w为1，再执行加法操作。

![image](https://user-images.githubusercontent.com/56336922/189571697-27231615-f728-4265-8521-b144e72f206e.png)

对`x==5，x==6`来说,它们的跳转表跳转到相同的代码块。

![image](https://user-images.githubusercontent.com/56336922/189571847-2bb3a0f0-b8a9-4ef7-abb9-42283e87bded.png)

对`x==4`来说，它将跳转到默认的代码块.L8中。
