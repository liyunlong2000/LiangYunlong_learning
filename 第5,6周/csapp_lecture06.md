# 程序级别编程 3：程序调用
程序调用主要涉及以下三个方面的实现机制：
- 传递控制：包括如何开始执行代码，以及如何返回到调用点
- 传递数据：包括程序需要的参数以及程序的返回值
- 内存管理：如何在程序执行的时候分配内存，以及在返回之后释放内存
## 栈结构
### x86-64栈
![image](https://user-images.githubusercontent.com/56336922/189573003-f65f55a6-06f8-4a82-9cfc-6c5f01106030.png)

- X64中栈是内存中一块区域，满足数据结构中栈先进后出的原则
- 栈底为高地址，栈向低地址方向生长
- 寄存器`%rsp`中地址指向了栈顶元素

### x86-64栈push
![image](https://user-images.githubusercontent.com/56336922/189573528-3565ca36-fbaf-49e3-8d36-95bac001d681.png)

- 首先将`%rsp`中地址减去8字节
- 将操作数写入`%rsp`指向的地址中

### x86-64栈pop
![image](https://user-images.githubusercontent.com/56336922/189573684-3860c9f6-88b2-4c8b-9c27-041ac3142d92.png)

与push操作相反：
- 首先根据`%rsp`中地址读出栈顶元素
- 再将`%rsp`中地址加上8字节

## 控制传递
### 程序控制流
- 通过栈实现程序的调用和返回
- 程序调用通过`call label`指令
  - 先将返回地址压入栈中
      - 返回地址指当前`call label`指令的下一条指令的地址
  - 跳转到指定标签
- 程序返回通过`ret`指令
  - 先将返回地址从栈中读出
  - 跳转到返回地址所指的指令中

### 控制流例子
mulstore函数将调用mult2函数。

![image](https://user-images.githubusercontent.com/56336922/189574509-a8bd29cf-65ae-47e7-8223-f06cc848d57f.png)

- `%rsp`中为栈顶地址0x120
- `%rip`中为当前指令的地址

![image](https://user-images.githubusercontent.com/56336922/189574772-15ba5b69-182c-430c-b6a2-14a21421563d.png)

当执行`call`指令时：
- 先将返回地址压入栈中
  - `%rsp`中地址减8,变为0x118
  - 返回地址为`call`指令的下一条指令的地址-0x400549，压入栈顶中
- 将`%rip`中当前指令的地址设置为所调用函数中指令的起始地址0x400550

![image](https://user-images.githubusercontent.com/56336922/189575187-053bfd9b-ae5e-4af4-bff1-dfd6c549f428.png)

当执行mult2中`ret`指令时：
- 取出栈中返回地址0x400549,移动至`%rip`中
- 将返回地址弹出栈中，`%rsp`中地址加8,变为0x120

![image](https://user-images.githubusercontent.com/56336922/189575544-7cd2aba8-8084-462a-8bf4-7048ca03de4f.png)

- `%rip`中当前指令的地址则为返回地址，控制流回到调用函数中
- mult2函数的返回值存放至`%rax`中，multstore函数可使用其返回值

### 程序数据流
![image](https://user-images.githubusercontent.com/56336922/189575821-ea26f224-978a-43cc-9545-cf534a8cdd58.png)

- 函数的前6个参数将依次存放至%rdi,...,%r9中
- 函数的返回值存放至%rax中
- 当必要时，多余的参数将存放至栈中

### 数据流例子
![image](https://user-images.githubusercontent.com/56336922/189576051-885670ad-dc1d-4780-9942-20e2c8c661a0.png)

- multstore中参数x，y，dest将依次存放至%rdi，%rsi，%rdx中
- mult2中参数a，b将依次存放至%rdi，%rsi中
- mult2的返回值(即t)将存放至%rax中，即t

## 局部数据管理
### 栈帧
栈帧包含以下内容：
- 返回信息
- 本地存储（如果需要）
- 临时空间（如果需要）
整一帧会在函数调用的时候进行空间分配，然后在返回时进行回收 

### x86-64/Linux 中栈帧
![image](https://user-images.githubusercontent.com/56336922/189577450-fd538f8a-f830-4bb1-9ee2-ca61a3f73a98.png)

当前要执行函数的栈帧中包含：
- 老栈帧的指针(可选)
- 本地变量
- 保存的寄存器上下文
- Argument build(当传递的参数多于6时，将存储在栈中)


调用者栈帧中包含：
- 返回地址(使用`call`指令压出栈中)
- 函数调用时的参数

### 数据管理例子
call_incr函数将调用incr函数。
![image](https://user-images.githubusercontent.com/56336922/189579464-7849f3b1-bad8-4729-b1f5-06e01a5343dd.png)

- 首先将`%rsp`中值减去16，相当于开了一块16字节的空间
- 将需要传递的参数15213存储至`%rsp+8`中地址所指内存

![image](https://user-images.githubusercontent.com/56336922/189579953-0bc16c5e-c3b7-45dd-889b-7f6ce4a9a43c.png)

- 接着将incr所需参数分别传入`%rdi`,`%rsi`中

![image](https://user-images.githubusercontent.com/56336922/189580194-f7459898-5cf5-4953-bf91-3606b94f3d11.png)

- 调用函数incr，`%rdi`中地址所指内存值将加上3000
- 返回值v2将存放至`%rax`中

![image](https://user-images.githubusercontent.com/56336922/189580419-26a7789f-6992-48e4-9f1d-82950a295f02.png)

- 将`%rsp+8`中地址所指内存值加到`%rax`中，即为当前函数返回值
- `%rsp`中值加上16，相当于回收了一块16字节的空间

![image](https://user-images.githubusercontent.com/56336922/189580733-8fde0bfc-c223-40ef-aa54-bfb7cf7256bf.png)

- 执行ret指令，返回到调用点

### 寄存器保存规定
- 调用者保存：当调用者调用函数时，需提前将临时值存储在自己栈帧中
- 被调用者保存：当被调用者要使用寄存器时，需提前将临时值存储在自己栈帧中，并且在返回时将临时值移动到寄存器中

### 被调用者保存例子
被调用者将使用寄存器%rbx临时存放参数x
![image](https://user-images.githubusercontent.com/56336922/189581608-7ad75000-00a1-47e9-b329-f8c3f3ad03e7.png)

- 首先%rbx中值存储至栈中
- 在开辟16字节大小的栈空间
- 将x移动到%rbx中

![image](https://user-images.githubusercontent.com/56336922/189582179-12ce334f-1815-4ef4-af98-3c6a57b8ce76.png)

- 回收16字节的栈空间
- 将栈中保存的%rbx中值弹回%rbx中

## 递归调用例子
### 递归函数的终止
![image](https://user-images.githubusercontent.com/56336922/189582540-2c336b87-714e-489e-be52-f5aad7d129aa.png)

- 通过je指令比较参数x和0的大小，判断是否要跳转到终止块

### 递归函数的寄存器保存
![image](https://user-images.githubusercontent.com/56336922/189582742-1c173b73-f01b-4a2a-a32f-5d11d73650fa.png)

- 当前函数将使用寄存器%rbx，因此先将%rbx中值保存至栈中，当函数返回时，在从栈中将值恢复。

### 递归函数的调用设置
![image](https://user-images.githubusercontent.com/56336922/189583085-97617ca7-b4f8-4f71-aeb9-17c033b129ed.png)

- 将调用函数所需的参数分别存放至%rdi
- %rbx存放当前函数返回所需的值`x & 1`属于被调用者保存

### 递归函数的调用
![image](https://user-images.githubusercontent.com/56336922/189583481-7083afc5-bd1c-4a9f-b672-fdd4855efed9.png)

- 所调用的函数返回结果存储在%rax中
- 被调用者将恢复%rbx中值，其值为与当前函数参数相关的`x & 1`

### 递归函数的结果

![image](https://user-images.githubusercontent.com/56336922/189583831-f6d7c5ce-009f-4d59-862e-efe65f48d31b.png)

- 将%rbx中值加到%rax中，即为当前函数的返回值

### 递归函数的完成
![image](https://user-images.githubusercontent.com/56336922/189583942-c44b33fb-e14f-41b6-b488-8b7dd2b1cdbe.png)

- 遵循被调用者保存规定，从栈中恢复%rbx中值


