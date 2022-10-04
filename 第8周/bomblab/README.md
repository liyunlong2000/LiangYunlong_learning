# Bomblab
## 实验说明
炸弹实验共有6阶段，每一阶段从终端读入一行字符串，然后根据`phase_*(input)`函数判断输入是否满足一定条件。若输入符合要求，则该阶段
保险解除，当6个阶段的保险都被解除时，炸弹则成功拆解。

bomblab目录下包含:
- bomb:可执行文件
- bomb.c：bomb的主函数
- README：实验介绍
- my_bomb.txt：对可执行文件反汇编得到的文本，包含个人注释
- gdbnotes-x86-64.pdf：gdb使用说明
## 实验准备
执行以下命令，对可执行文件进行反汇编，得到炸弹程序的汇编码。
```
objdump -d bomb > my_bomb.txt
```
## phase_1
phase_1中关键汇编代码如下：
```
  400ee4:	be 00 24 40 00       	mov    $0x402400,%esi
  400ee9:	e8 4a 04 00 00       	callq  401338 <strings_not_equal>
  400eee:	85 c0                	test   %eax,%eax
  400ef0:	74 05                	je     400ef7 <phase_1+0x17>
  400ef2:	e8 43 05 00 00       	callq  40143a <explode_bomb>
  400ef7:	48 83 c4 08          	add    $0x8,%rsp
```
执行逻辑为：将常量地址为`$0x402400`的字符串和输入的字符串(存储在`%rdi`中)作为函数`strings_not_equal`的参数，进行字符串
比较，不相等时则引爆炸弹。

依次执行以下命令，查看常量地址为`$0x402400`的字符串。
```
//调试可执行程序bomb
gdb bomb
//查看字符串内容
x/s 0x402400
```
输出结果如下：
```
0x402400:       "Border relations with Canada have never been better."
```
因此，我们应将以上内容作为阶段1的输入,即:
```
Border relations with Canada have never been better.
```
## phase_2
phase_2中关键汇编码为：
```
  400f05:	e8 52 05 00 00       	callq  40145c <read_six_numbers>  //调用函数，读入六个数字
  400f0a:	83 3c 24 01          	cmpl   $0x1,(%rsp)                //比较%rsp中地址所指内存中值和$0x1中值
  400f0e:	74 20                	je     400f30 <phase_2+0x34>      //相等则跳转到400f30
  400f10:	e8 25 05 00 00       	callq  40143a <explode_bomb>      //不相等则爆炸
  400f15:	eb 19                	jmp    400f30 <phase_2+0x34>      //跳转到400f30
  400f17:	8b 43 fc             	mov    -0x4(%rbx),%eax            //将地址rbx-4中地址所指值存入eax中    
  400f1a:	01 c0                	add    %eax,%eax                  //eax中值乘2
  400f1c:	39 03                	cmp    %eax,(%rbx)                //比较rbx中地址所指向的值和eax中值
  400f1e:	74 05                	je     400f25 <phase_2+0x29>      //相等则跳转到400f25
  400f20:	e8 15 05 00 00       	callq  40143a <explode_bomb>      //不相等则引爆炸弹
  400f25:	48 83 c3 04          	add    $0x4,%rbx                  //rbx中值加4
  400f29:	48 39 eb             	cmp    %rbp,%rbx                  //比较rbx和rbp
  400f2c:	75 e9                	jne    400f17 <phase_2+0x1b>      //不相等跳转到400f17
  400f2e:	eb 0c                	jmp    400f3c <phase_2+0x40>      //跳转到400f3c
  400f30:	48 8d 5c 24 04       	lea    0x4(%rsp),%rbx             //将地址rsp+4存入rbx中
  400f35:	48 8d 6c 24 18       	lea    0x18(%rsp),%rbp            //将地址rsp+18存入rbp中
  400f3a:	eb db                	jmp    400f17 <phase_2+0x1b>      //跳转到400f17
  ```
  首先调用函数`read_six_numbers`,读入六个整型数，首地址为`%rsp`.然后比较第一个数和1,若不相等则引爆炸弹.后续为循环，循环终止条件为`%rbp==%rbx`,循环变量为`%rbx`,起始条件
  为`%rbx=%rsp+4`,变化条件为`%rbx=%rbx+4`,循环体中比较`(%rbx)`是否与`2*(%rbx-4)`相等，不相等则引爆.
  
  因此，阶段2的字符串中第一个数为1，后续的每个数都为前一个数的两倍.即
  ```
  1 2 4 8 16 32
  ```
  
  ## phase_3
  阶段3中初始汇编代码为:
  ```
  400f47:	48 8d 4c 24 0c       	lea    0xc(%rsp),%rcx
  400f4c:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  400f51:	be cf 25 40 00       	mov    $0x4025cf,%esi
  400f56:	b8 00 00 00 00       	mov    $0x0,%eax
  400f5b:	e8 90 fc ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  ```
  首先调用函数`sscanf()`,函数需要一个格式化字符串,其在地址`$0x4025cf`中,执行以下命令查看格式化字符串:
  ```
  x/s 0x4025cf
  ```
  输出结果为：
  ```
  0x4025cf:       "%d %d"
  ```
  因此,函数`sscanf()`共有四个参数：
  - %rdi：输入的字符串
  - %esi：格式化字符串
  - %rdx：读入的一个数字的地址
  - %rcx：读入的二个数字的地址
  
  phase_3中关键汇编代码如下：
 ```
  400f71:	8b 44 24 08          	mov    0x8(%rsp),%eax           //将rsp+8中地址所指内存值(rdx)存入eax中
  400f75:	ff 24 c5 70 24 40 00 	jmpq   *0x402470(,%rax,8)
  400f7c:	b8 cf 00 00 00       	mov    $0xcf,%eax
  400f81:	eb 3b                	jmp    400fbe <phase_3+0x7b>
  400f83:	b8 c3 02 00 00       	mov    $0x2c3,%eax
  400f88:	eb 34                	jmp    400fbe <phase_3+0x7b>
  400f8a:	b8 00 01 00 00       	mov    $0x100,%eax
  400f8f:	eb 2d                	jmp    400fbe <phase_3+0x7b>
  400f91:	b8 85 01 00 00       	mov    $0x185,%eax
  400f96:	eb 26                	jmp    400fbe <phase_3+0x7b>
  400f98:	b8 ce 00 00 00       	mov    $0xce,%eax
  400f9d:	eb 1f                	jmp    400fbe <phase_3+0x7b>
  400f9f:	b8 aa 02 00 00       	mov    $0x2aa,%eax
  400fa4:	eb 18                	jmp    400fbe <phase_3+0x7b>
  400fa6:	b8 47 01 00 00       	mov    $0x147,%eax
  400fab:	eb 11                	jmp    400fbe <phase_3+0x7b>
 .....
  400fbe:	3b 44 24 0c          	cmp    0xc(%rsp),%eax         //比较eax和rsp+c中地址所指内存值的大小
  400fc2:	74 05                	je     400fc9 <phase_3+0x86>  //相等则跳转到400fc9,结束
  ```
  根据读入索引`%rdx`,根据跳转表(在地址0x402470中)中具体地址进行跳转.跳转到的各个case中,将具体数值存入`%eax`中.最后
 ,将该值与读入的第二个数字进行对比，不相等则引爆炸弹.
 
 执行以下命令查看跳转表的实际的跳转地址：
 ```
x/8a 0x402470
 ```
 输出结果为：
 ```
0x402470:       0x400f7c <phase_3+57>   0x400fb9 <phase_3+118>
0x402480:       0x400f83 <phase_3+64>   0x400f8a <phase_3+71>
0x402490:       0x400f91 <phase_3+78>   0x400f98 <phase_3+85>
0x4024a0:       0x400f9f <phase_3+92>   0x400fa6 <phase_3+99>
 ```
 根据以上内容可用得出实际跳转的跳转地址,例如索引为0时跳转到`0x400f7c`，索引为1时跳转到`0x400fb9`...,这里选择索引0,实际跳转
 的块中将`0xcf(即207)`赋值给`%eax`.
 
 因此，阶段3中可行的答案为：
 ```
 0 207
 ```
 ## phase_4
 阶段4的初始部分同阶段3:
 ```
  401010:	48 8d 4c 24 0c       	lea    0xc(%rsp),%rcx
  401015:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  40101a:	be cf 25 40 00       	mov    $0x4025cf,%esi
  40101f:	b8 00 00 00 00       	mov    $0x0,%eax
  401024:	e8 c7 fb ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  401029:	83 f8 02             	cmp    $0x2,%eax
  40102c:	75 07                	jne    401035 <phase_4+0x29>   //eax中值不等于2则爆炸，即未读入两个数
  ```
  首先读入两个数,分别存储在`%rdx`,`%rcx`中.
  ```
  40102e:	83 7c 24 08 0e       	cmpl   $0xe,0x8(%rsp)         //比较rdx与14，第一个参数
  401033:	76 05                	jbe    40103a <phase_4+0x2e>  //小于等于14则跳转到40103a
  401035:	e8 00 04 00 00       	callq  40143a <explode_bomb>  //大于14则爆炸
  40103a:	ba 0e 00 00 00       	mov    $0xe,%edx              //将rdx设置为14
  40103f:	be 00 00 00 00       	mov    $0x0,%esi              //将esi设置为0
  ```
  然后第一个参数进行判断,其必须小于等于14.
  ```
  40103a:	ba 0e 00 00 00       	mov    $0xe,%edx              //将rdx设置为14
  40103f:	be 00 00 00 00       	mov    $0x0,%esi              //将esi设置为0
  401044:	8b 7c 24 08          	mov    0x8(%rsp),%edi         //将读入的第一个参数存入edi
  401048:	e8 81 ff ff ff       	callq  400fce <func4>         //调用函数func4
  40104d:	85 c0                	test   %eax,%eax
  40104f:	75 07                	jne    401058 <phase_4+0x4c>  //返回值不为0则爆炸
  401051:	83 7c 24 0c 00       	cmpl   $0x0,0xc(%rsp)         //比较第二个参数与0的大小
  401056:	74 05                	je     40105d <phase_4+0x51>  //相等则结束
  401058:	e8 dd 03 00 00       	callq  40143a <explode_bomb>  //不相等则引爆炸弹
  ```
  之后将调用函数`func4`,三个参数分别为`%edi(读入的第一个数)`,`%esi(=0)`,`%edx(=14)`,函数返回值存储在
  `%rax`中，若其不为0则引爆炸弹,并且后续读入的第二个参数必须为0，否则也会引爆炸弹.因此,输出的第一个数字
  应使函数`func4`的返回值为0.
  
  函数`fun4`中关键汇编代码如下：
  ```
  400fd2:	89 d0                	mov    %edx,%eax            //将14存入eax
  400fd4:	29 f0                	sub    %esi,%eax            //eax=14-0；
  400fd6:	89 c1                	mov    %eax,%ecx            //ecx=eax
  400fd8:	c1 e9 1f             	shr    $0x1f,%ecx           //ecx中数逻辑右移31位
  400fdb:	01 c8                	add    %ecx,%eax            //eax=eax+0=14
  400fdd:	d1 f8                	sar    %eax                 //eax算术右移一位，eax=7
  400fdf:	8d 0c 30             	lea    (%rax,%rsi,1),%ecx   //ecx=rax+rsi=7+0=7;
  400fe2:	39 f9                	cmp    %edi,%ecx            //比较ecx和edi，即7和第一个参数
  400fe4:	7e 0c                	jle    400ff2 <func4+0x24>  //第一个参数大于等于7则跳转400ff2
  400fe6:	8d 51 ff             	lea    -0x1(%rcx),%edx
  400fe9:	e8 e0 ff ff ff       	callq  400fce <func4>
  400fee:	01 c0                	add    %eax,%eax
  400ff0:	eb 15                	jmp    401007 <func4+0x39>
  400ff2:	b8 00 00 00 00       	mov    $0x0,%eax            //eax置为0
  400ff7:	39 f9                	cmp    %edi,%ecx            //比较ecx和edi，即7和第一个参数
  400ff9:	7d 0c                	jge    401007 <func4+0x39>  //7大于等于第一个参数则跳转到401007，结束
  ```
  这是一个递归函数，但是仔细观察可以发现一条简单的执行路径,即当输入参数为7时.此时不需要递归调用，并且返回值为0.
  因此,阶段4的一个可行的答案为:
  ```
  7 0
  ```
  ## phase_5
  阶段5初始部分汇编码为:
  ```
  40107a:	e8 9c 02 00 00       	callq  40131b <string_length>   //调动string_length
  40107f:	83 f8 06             	cmp    $0x6,%eax                //比较字符串长度和6
  401082:	74 4e                	je     4010d2 <phase_5+0x70>    //相等则跳转到4010d2
  401084:	e8 b1 03 00 00       	callq  40143a <explode_bomb>    //不相等则爆炸
  401089:	eb 47                	jmp    4010d2 <phase_5+0x70>    //跳转到4010d2
  ```
  对输入字符串长度进行判断,当长度不为6时则引爆炸弹.
  
  阶段5的关键汇编代码如下：
  ```
  40108b:	0f b6 0c 03          	movzbl (%rbx,%rax,1),%ecx     //ecx=rbx+rax
  40108f:	88 0c 24             	mov    %cl,(%rsp)             //rsp中值=cl
  401092:	48 8b 14 24          	mov    (%rsp),%rdx            //edx=rsp中值
  401096:	83 e2 0f             	and    $0xf,%edx              //去edx的后四位
  401099:	0f b6 92 b0 24 40 00 	movzbl 0x4024b0(%rdx),%edx    //edx=rdx+0x4024b0
  4010a0:	88 54 04 10          	mov    %dl,0x10(%rsp,%rax,1)  //*(rsp+rax+0x10) = dl
  4010a4:	48 83 c0 01          	add    $0x1,%rax              //rax+=1
  4010a8:	48 83 f8 06          	cmp    $0x6,%rax              //rax与6比大小
  4010ac:	75 dd                	jne    40108b <phase_5+0x29>  //不相等则跳转40108b
  4010ae:	c6 44 24 16 00       	movb   $0x0,0x16(%rsp)        //相等则*(rsp + 0x16) = 0;
  4010b3:	be 5e 24 40 00       	mov    $0x40245e,%esi         //esi = 0x40245e;
  4010b8:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi        //rdi = rsp + 10;
  4010bd:	e8 76 02 00 00       	callq  401338 <strings_not_equal> //调用strings_not_equal
  4010c2:	85 c0                	test   %eax,%eax              //测试eax是否为0 
  4010c4:	74 13                	je     4010d9 <phase_5+0x77>  //为0则跳转到4010d9
  ```
  首先是一个循环,循环变量为`%rax`,变量初始化为`%rax=0`,循环条件为`%rax!=6`,变化条件为`%rax=%rax+1`.
  在循环体中,`%rbx`为输入字符串的地址,每次取出输入字符串的一个字符,将其值与上`$0xf`,即保留后四位二进制数,然后将其作为偏移地址,将常量地址为`0x4024b0`的字符串的中相应
  字符存回输入字符串中.最后将其与常量地址为`0x40245e`的字符串进行比较,若不相等则引爆炸弹.
  
  执行以下命令,查看常量地址为`0x40245e`的字符串.
  ```
  x/s 0x40245e
  ```
  输出为:
  ```
  0x40245e:       "flyers"
  ```
  执行以下命令,查看常量地址为`0x4024b0`的字符串.
  ```
  x/s 0x4024b0
  ```
  输出为:
  ```
  0x4024b0 <array.3449>:  "maduiersnfotvbylSo you think you can stop the bomb with ctrl-c, do you?"
  ```
  为了使最后的输入字符串与"flyers"相等,我们应在常量地址为`0x4024b0`的字符串找到相应字符的偏移地址,根据该地址进行输入.字符串"flyers"中各字符的偏移地址为9,15,14,5,6,7.
  因此,我们输入的6个字符应该满足：每个字符的ASCII码后四位应该为对应的数字(可在16进制的ASCII码中快速找到).一个可行的答案为：
  ```
  ionefg
  ```
  
  
