# 机器语言编程基础
## 英特尔处理器和架构
- x86泛指一系列基于Intel 8086且向后兼容的中央处理器指令集架构
- Intel在早期以80x86这样的数字格式来命名处理器，包括Intel 8086、80186、80286、80386以及80486，由于以“86”作为结尾，因此其架构被称为“x86”
- X86架构是CISC（复杂指令集计算机，Complex Instruction Set Computer）
## C、汇编和机器码
### 定义&概念
- 架构(指令集架构)：一个能为电路硬件翻译应用程序的一层抽象层。
  - 抽象出操作系统中规则和约束，例如：特定的指令集(x86,x86-64,IA32),寄存器有多少个。
- 微架构：架构的具体实现。
  - 例如：缓存器的大小、主频
- 机器代码形式分为：
  - 机器码：处理器实际执行的字节版本的程序
  - 汇编代码：机器码的文本表示
### 汇编/机器码视图
![image](https://user-images.githubusercontent.com/56336922/189513821-80fdc468-a6fe-4af7-8f5b-79d23f6cf981.png)
从汇编/机器码编程者的角度来看，可见如下状态信息：
- PC(程序计数器):下一条指令的地址，在x86-64中叫RIP
- 寄存器：存储数据
- 条件代码：保存最近的算术或逻辑操作的状态信息，用来实现条件分支
- 内存：看作一个可寻址的字节数组，存放代码和用户数据，包含程序运行所需的栈区。
### 从C到目标代码
![image](https://user-images.githubusercontent.com/56336922/189514127-a296a2cc-a42c-4eca-94ae-ed3785577913.png)
1. c代码通过编译器处理，成为汇编代码
2. 汇编代码通过汇编器处理，将文本表示的汇编代码转换为实际执行的机器码
3. 机器码和静态库通过链接器的处理，成为可执行程序
### 机器指令例子
![image](https://user-images.githubusercontent.com/56336922/189514413-2dfe0465-3b03-40a4-95b8-c6e3c2727068.png)
- C代码：
  - 将操作数t的值存放在dest中地址所指的内存中
- 汇编代码：
  - 操作数t和dest分别存在寄存器%rax、%rbx中，其中dest是一个地址
  - (%rbx)是间接寻址，表示将%rax中值存在%rbx中地址所指的内存中
- 目标代码：
  - 48，89，03说明这是3字节的指令，是指令的具体字节表示。
  - 该指令存储在地址0x40059e中
## 汇编基础：寄存器、操作数、移动
### X86-64中整型寄存器
![image](https://user-images.githubusercontent.com/56336922/189514780-8ff039ed-f322-4105-8e10-618ed3759b61.png)

- 有16个64位的寄存器
- 可以使用%e**或%r**d使用寄存器的低32位
- 甚至还可以引用低8位或16位
  - 这也是x86-64能够向下兼容的原因
- %rsp：作为栈指针使用
### 数据移动
```
movq Source，Dest
```
上述指令将源操作数移动到目的操作数。操作数分为：
- 立即数：常整数值
  - 例如：$0x400,$-533,需要以$为前缀
- 寄存器：16个寄存器中值
  - 例如：%ras、%r13，注意 %rsp保留作为栈指针使用
- 内存：需要通过8字节地址访问内存中值
  - 例如：(%rax)表示寄存器%rax中地址所指内存中的值

![image](https://user-images.githubusercontent.com/56336922/189515129-220f9bfe-8481-48aa-b390-b6a5be620aec.png)
对于movq指令，源操作数和目标操作数可以是以上操作数的组合，但是一条指令无法实现内存到内存之间数据移动，所以只有5种可能的组合。

### 内存访问模式例子
![image](https://user-images.githubusercontent.com/56336922/189515218-ac7b8d7d-8578-4a3a-991d-8d99324d89aa.png)
- 使用了如movq (%rcx),%rax形式，内存地址为%rcx中值

![image](https://user-images.githubusercontent.com/56336922/189515351-b96ff600-2aa3-4308-9b42-67c6a32aa160.png)
- 使用了一般形式D(Rb，Ri，S)，内存地址为Reg [ Rb ] +S* Reg[ Ri ] +	D
## 算术和逻辑操作
### 地址计算指令
```
leaq Source，Dest
```
上述指令取出Source中操作数的有效地址，将其存放至Dest中。具体用法有两种：
1. 无需通过内存引用而计算出地址
2. 计算形如x+k * y形式的表达式，其中k=1,2,4,8.

![image](https://user-images.githubusercontent.com/56336922/189515590-4c716f60-d73e-4c6f-bc8e-e3023a5907af.png)
上述例子通过leaq和salq指令实现了x * 12，%rdx中值为x，这里lea将其当做地址进行运算。

### 算术表达式例子
![image](https://user-images.githubusercontent.com/56336922/189515662-746d691d-d07f-4a26-b873-f393e757d18b.png)

程序刚执行时：
- %rdi存放参数x
- %rsi存放参数y
- %rdx存放参数z

```
leaq (%rdi,%rsi), %rax
```
将内存地址为x+y的值的有效地址移动到%rax中(即x+y)，此时%rax为t1

```
addq %rdx, %rax
```
将z与%rax中值t1相加并存放至%rax中，此时%rax为t2


```
leaq (%rsi,%rsi,2), %rdx
salq $4, %rdx
```
将3倍%rsi中值y存放至%rdx中，此时%rdx为3 * y。然后，在将%rdx中值左移四位，得到48 * y，此时%rdx中值为t4.

```
leaq 4(%rdi,%rdx), %rcx
```
此时%rdi中值为x，%rdx为t4。将x+4+t4的值存放至%rcx中，即t5。

```
imulq %rcx, %rax 
ret 
```
将%rcx中t5和%rax中t2相乘，并存放至%rax中，即rval。最后通过ret指令返回，%rax保存程序的返回值。






