# 机器级别编程 4：数据
## 数组
### 数组分配
![image](https://user-images.githubusercontent.com/56336922/189606939-fcc7a8c6-a617-449a-aa4e-c797961578a2.png)

- 对于类型为T，长度为L的数组`T A[L]`,分配连续的大小为`L * sizeof(T)`字节的内存区域

### 数组访问
![image](https://user-images.githubusercontent.com/56336922/189607275-6b81acf8-d539-4443-989e-aeb02833c542.png)

- 数组的标识符val作为数组首元素的指针
![image](https://user-images.githubusercontent.com/56336922/189607456-b2f6865c-b85c-4b96-9545-16faccb9d4c9.png)

- 数组标识符val的加法，被加数需要乘以比例系数`sizeof(T)`
- 通过`&`取地址
- 通过`*`解引用

### 数组例子
![image](https://user-images.githubusercontent.com/56336922/189608443-e5c26682-8f45-475f-a035-9acb258685dd.png)

- 使用`typedef`为数组定义简洁的名称，这里`zip_dig cmu`相当于`int cmu[5]`

### 数组访问例子
![image](https://user-images.githubusercontent.com/56336922/189608937-875d5ebe-84d3-4f13-87b2-5cd1858e36b1.png)

- `%rdi`中值为数组首元素的地址，`%rsi`中值为要访问元素的索引
- 所访问元素的有效地址为`%rdi + 4*%rsi`，需要乘以比例系数4，即int类型的大小

### 多维数组
![image](https://user-images.githubusercontent.com/56336922/189610102-e5701943-ec4d-439d-8a3e-d8126981b975.png)

-  对于类型为T，长度为`R*L`的数组`T A[R][C]`,分配连续的大小为`R*C*sizeof(T)`字节的内存区域
-  这里采用行优先顺序，相当于按行遍历，依次将矩阵元素存放至内存区域中

### 多维数组例子
![image](https://user-images.githubusercontent.com/56336922/189610349-4e4e0e82-297b-4540-b828-b753c073ad19.png)

- 由于前面使用了`typedef int zip_dig[5]`，这里`zip_dig pgh[4]`相当于`int pgh[4][5]`
- 分配连续的大小为`4*5*4`字节的内存区域
  - pgh是一个包含4个元素的数组，内存是连续分配的
  - pgh中每个元素是一个包含5个元素的数组，内存是连续分配的

### 多维数组访问
![image](https://user-images.githubusercontent.com/56336922/189612434-ef67ba45-b72b-4c37-b477-59d19928a2ea.png)


- `i*C*4`表示了元素所在的行数组的首元素地址
  - `A[i]`表示含有C个元素的数组
- `j*4`表示元素在行数组内的偏移地址


![image](https://user-images.githubusercontent.com/56336922/189613239-ade325c1-4e79-4f20-a2ba-aedef97e42ed.png)

- 因为数组每行有5个元素，首先计算`5*index`，得到元素所在行数组首元素的序号
- 再加上`dig`，得到元素的序号
- 最后乘以类型比例系数，得到元素的实际地址

### 多级阵列例子
![image](https://user-images.githubusercontent.com/56336922/189614897-c0d0a7b2-88c7-4679-a691-417dbe1f3825.png)

- univ是包含三个元素的数组
  - 数组元素类型为指针，每个元素是包含5个元素的数组首元素的地址
  - 指针大小为8字节
- 整个多级阵列的内存分配不是连续的 

### 多级阵列访问
![image](https://user-images.githubusercontent.com/56336922/189616129-d4407923-20fc-45b4-93a6-f465377b9aec.png)

- 首先计算行数组内的偏移地址`4*digit`
- 然后计算行数组的地址`univ(,%rdi,8)`,即为`univ+8*index`所指内存中的值
- 将它们相加得到元素的内存地址
- 解引用并将其返回

## 结构体
### 结构体表示
# 机器级别编程 4：数据
## 数组
### 数组分配
![image](https://user-images.githubusercontent.com/56336922/189606939-fcc7a8c6-a617-449a-aa4e-c797961578a2.png)

- 对于类型为T，长度为L的数组`T A[L]`,分配连续的大小为`L * sizeof(T)`字节的内存区域

### 数组访问
![image](https://user-images.githubusercontent.com/56336922/189607275-6b81acf8-d539-4443-989e-aeb02833c542.png)

- 数组的标识符val作为数组首元素的指针
![image](https://user-images.githubusercontent.com/56336922/189607456-b2f6865c-b85c-4b96-9545-16faccb9d4c9.png)

- 数组标识符val的加法，被加数需要乘以比例系数`sizeof(T)`
- 通过`&`取地址
- 通过`*`解引用

### 数组例子
![image](https://user-images.githubusercontent.com/56336922/189608443-e5c26682-8f45-475f-a035-9acb258685dd.png)

- 使用`typedef`为数组定义简洁的名称，这里`zip_dig cmu`相当于`int cmu[5]`

### 数组访问例子
![image](https://user-images.githubusercontent.com/56336922/189608937-875d5ebe-84d3-4f13-87b2-5cd1858e36b1.png)

- `%rdi`中值为数组首元素的地址，`%rsi`中值为要访问元素的索引
- 所访问元素的有效地址为`%rdi + 4*%rsi`，需要乘以比例系数4，即int类型的大小

### 多维数组
![image](https://user-images.githubusercontent.com/56336922/189610102-e5701943-ec4d-439d-8a3e-d8126981b975.png)

-  对于类型为T，长度为`R*L`的数组`T A[R][C]`,分配连续的大小为`R*C*sizeof(T)`字节的内存区域
-  这里采用行优先顺序，相当于按行遍历，依次将矩阵元素存放至内存区域中

### 多维数组例子
![image](https://user-images.githubusercontent.com/56336922/189610349-4e4e0e82-297b-4540-b828-b753c073ad19.png)

- 由于前面使用了`typedef int zip_dig[5]`，这里`zip_dig pgh[4]`相当于`int pgh[4][5]`
- 分配连续的大小为`4*5*4`字节的内存区域
  - pgh是一个包含4个元素的数组，内存是连续分配的
  - pgh中每个元素是一个包含5个元素的数组，内存是连续分配的

### 多维数组访问
![image](https://user-images.githubusercontent.com/56336922/189612434-ef67ba45-b72b-4c37-b477-59d19928a2ea.png)


- `i*C*4`表示了元素所在的行数组的首元素地址
  - `A[i]`表示含有C个元素的数组
- `j*4`表示元素在行数组内的偏移地址


![image](https://user-images.githubusercontent.com/56336922/189613239-ade325c1-4e79-4f20-a2ba-aedef97e42ed.png)

- 因为数组每行有5个元素，首先计算`5*index`，得到元素所在行数组首元素的序号
- 再加上`dig`，得到元素的序号
- 最后乘以类型比例系数，得到元素的实际地址

### 多级阵列例子
![image](https://user-images.githubusercontent.com/56336922/189614897-c0d0a7b2-88c7-4679-a691-417dbe1f3825.png)

- univ是包含三个元素的数组
  - 数组元素类型为指针，每个元素是包含5个元素的数组首元素的地址
  - 指针大小为8字节
- 整个多级阵列的内存分配不是连续的 

### 多级阵列访问
![image](https://user-images.githubusercontent.com/56336922/189616129-d4407923-20fc-45b4-93a6-f465377b9aec.png)

- 首先计算行数组内的偏移地址`4*digit`
- 然后计算行数组的地址`univ(,%rdi,8)`,即为`univ+8*index`所指内存中的值
- 将它们相加得到元素的内存地址
- 解引用并将其返回

## 结构体
### 结构体表示
# 机器级别编程 4：数据
## 数组
### 数组分配
![image](https://user-images.githubusercontent.com/56336922/189606939-fcc7a8c6-a617-449a-aa4e-c797961578a2.png)

- 对于类型为T，长度为L的数组`T A[L]`,分配连续的大小为`L * sizeof(T)`字节的内存区域

### 数组访问
![image](https://user-images.githubusercontent.com/56336922/189607275-6b81acf8-d539-4443-989e-aeb02833c542.png)

- 数组的标识符val作为数组首元素的指针
![image](https://user-images.githubusercontent.com/56336922/189607456-b2f6865c-b85c-4b96-9545-16faccb9d4c9.png)

- 数组标识符val的加法，被加数需要乘以比例系数`sizeof(T)`
- 通过`&`取地址
- 通过`*`解引用

### 数组例子
![image](https://user-images.githubusercontent.com/56336922/189608443-e5c26682-8f45-475f-a035-9acb258685dd.png)

- 使用`typedef`为数组定义简洁的名称，这里`zip_dig cmu`相当于`int cmu[5]`

### 数组访问例子
![image](https://user-images.githubusercontent.com/56336922/189608937-875d5ebe-84d3-4f13-87b2-5cd1858e36b1.png)

- `%rdi`中值为数组首元素的地址，`%rsi`中值为要访问元素的索引
- 所访问元素的有效地址为`%rdi + 4*%rsi`，需要乘以比例系数4，即int类型的大小

### 多维数组
![image](https://user-images.githubusercontent.com/56336922/189610102-e5701943-ec4d-439d-8a3e-d8126981b975.png)

-  对于类型为T，长度为`R*L`的数组`T A[R][C]`,分配连续的大小为`R*C*sizeof(T)`字节的内存区域
-  这里采用行优先顺序，相当于按行遍历，依次将矩阵元素存放至内存区域中

### 多维数组例子
![image](https://user-images.githubusercontent.com/56336922/189610349-4e4e0e82-297b-4540-b828-b753c073ad19.png)

- 由于前面使用了`typedef int zip_dig[5]`，这里`zip_dig pgh[4]`相当于`int pgh[4][5]`
- 分配连续的大小为`4*5*4`字节的内存区域
  - pgh是一个包含4个元素的数组，内存是连续分配的
  - pgh中每个元素是一个包含5个元素的数组，内存是连续分配的

### 多维数组访问
![image](https://user-images.githubusercontent.com/56336922/189612434-ef67ba45-b72b-4c37-b477-59d19928a2ea.png)


- `i*C*4`表示了元素所在的行数组的首元素地址
  - `A[i]`表示含有C个元素的数组
- `j*4`表示元素在行数组内的偏移地址


![image](https://user-images.githubusercontent.com/56336922/189613239-ade325c1-4e79-4f20-a2ba-aedef97e42ed.png)

- 因为数组每行有5个元素，首先计算`5*index`，得到元素所在行数组首元素的序号
- 再加上`dig`，得到元素的序号
- 最后乘以类型比例系数，得到元素的实际地址

### 多级阵列例子
![image](https://user-images.githubusercontent.com/56336922/189614897-c0d0a7b2-88c7-4679-a691-417dbe1f3825.png)

- univ是包含三个元素的数组
  - 数组元素类型为指针，每个元素是包含5个元素的数组首元素的地址
  - 指针大小为8字节
- 整个多级阵列的内存分配不是连续的 

### 多级阵列访问
![image](https://user-images.githubusercontent.com/56336922/189616129-d4407923-20fc-45b4-93a6-f465377b9aec.png)

- 首先计算行数组内的偏移地址`4*digit`
- 然后计算行数组的地址`univ(,%rdi,8)`,即为`univ+8*index`所指内存中的值
- 将它们相加得到元素的内存地址
- 解引用并将其返回

## 结构体
### 结构体表示
![image](https://user-images.githubusercontent.com/56336922/189619954-40e0e515-b92d-4712-bb89-d01af5cd4fff.png)

- 为结构体每个字段分配足够的内存大小
- 字段的顺序取决于声明的顺序




- 为结构体每个字段分配足够的内存大小
- 字段的顺序取决于声明的顺序




# 机器级别编程 4：数据
## 数组
### 数组分配
![image](https://user-images.githubusercontent.com/56336922/189606939-fcc7a8c6-a617-449a-aa4e-c797961578a2.png)

- 对于类型为T，长度为L的数组`T A[L]`,分配连续的大小为`L * sizeof(T)`字节的内存区域

### 数组访问
![image](https://user-images.githubusercontent.com/56336922/189607275-6b81acf8-d539-4443-989e-aeb02833c542.png)

- 数组的标识符val作为数组首元素的指针
![image](https://user-images.githubusercontent.com/56336922/189607456-b2f6865c-b85c-4b96-9545-16faccb9d4c9.png)

- 数组标识符val的加法，被加数需要乘以比例系数`sizeof(T)`
- 通过`&`取地址
- 通过`*`解引用

### 数组例子
![image](https://user-images.githubusercontent.com/56336922/189608443-e5c26682-8f45-475f-a035-9acb258685dd.png)

- 使用`typedef`为数组定义简洁的名称，这里`zip_dig cmu`相当于`int cmu[5]`

### 数组访问例子
![image](https://user-images.githubusercontent.com/56336922/189608937-875d5ebe-84d3-4f13-87b2-5cd1858e36b1.png)

- `%rdi`中值为数组首元素的地址，`%rsi`中值为要访问元素的索引
- 所访问元素的有效地址为`%rdi + 4*%rsi`，需要乘以比例系数4，即int类型的大小

### 多维数组
![image](https://user-images.githubusercontent.com/56336922/189610102-e5701943-ec4d-439d-8a3e-d8126981b975.png)

-  对于类型为T，长度为`R*L`的数组`T A[R][C]`,分配连续的大小为`R*C*sizeof(T)`字节的内存区域
-  这里采用行优先顺序，相当于按行遍历，依次将矩阵元素存放至内存区域中

### 多维数组例子
![image](https://user-images.githubusercontent.com/56336922/189610349-4e4e0e82-297b-4540-b828-b753c073ad19.png)

- 由于前面使用了`typedef int zip_dig[5]`，这里`zip_dig pgh[4]`相当于`int pgh[4][5]`
- 分配连续的大小为`4*5*4`字节的内存区域
  - pgh是一个包含4个元素的数组，内存是连续分配的
  - pgh中每个元素是一个包含5个元素的数组，内存是连续分配的

### 多维数组访问
![image](https://user-images.githubusercontent.com/56336922/189612434-ef67ba45-b72b-4c37-b477-59d19928a2ea.png)


- `i*C*4`表示了元素所在的行数组的首元素地址
  - `A[i]`表示含有C个元素的数组
- `j*4`表示元素在行数组内的偏移地址


![image](https://user-images.githubusercontent.com/56336922/189613239-ade325c1-4e79-4f20-a2ba-aedef97e42ed.png)

- 因为数组每行有5个元素，首先计算`5*index`，得到元素所在行数组首元素的序号
- 再加上`dig`，得到元素的序号
- 最后乘以类型比例系数，得到元素的实际地址

### 多级阵列例子
![image](https://user-images.githubusercontent.com/56336922/189614897-c0d0a7b2-88c7-4679-a691-417dbe1f3825.png)

- univ是包含三个元素的数组
  - 数组元素类型为指针，每个元素是包含5个元素的数组首元素的地址
  - 指针大小为8字节
- 整个多级阵列的内存分配不是连续的 

### 多级阵列访问
![image](https://user-images.githubusercontent.com/56336922/189616129-d4407923-20fc-45b4-93a6-f465377b9aec.png)

- 首先计算行数组内的偏移地址`4*digit`
- 然后计算行数组的地址`univ(,%rdi,8)`,即为`univ+8*index`所指内存中的值
- 将它们相加得到元素的内存地址
- 解引用并将其返回

## 结构体
### 结构体表示
![image](https://user-images.githubusercontent.com/56336922/189620360-1db579e5-b196-4ca0-92ce-8ebe31dfd008.png)

- 为结构体每个字段分配足够的内存大小
- 字段的顺序取决于声明的顺序


### 生成结构体成员的指针
![image](https://user-images.githubusercontent.com/56336922/189620403-f48203a6-b0d1-4c93-9a51-5345dcff3b4b.png)

- 结构体标识符为首元素的地址
- 编译时决定结构体成员的偏移，根据偏移即可得到各成员的地址

### 链表
![image](https://user-images.githubusercontent.com/56336922/189620870-4a4754f8-6e85-4a28-afd8-d60bc3218706.png)

- 首先取出结构体中i，偏移为16字节，基地址为`%rdi`
- 然后访问结构体中元素`r->a[i]`,比例系数为4,偏移为`4*i`字节，基地址为`%rdi`
- 最后访问`r->next`，偏移为24字节，基地址为`%rdi`

### 对齐属性
![image](https://user-images.githubusercontent.com/56336922/189622085-97746e69-4069-4257-83f0-4dd221f8ca8b.png)

- 数据对齐原则：对于K字节的原始数据类型，地址必须是K的倍数
- 数据对齐好处：内存访问通常是4或者8个字节位单位的，若数据跨越两个区域，这样会导致访问效率降低
- 编译器会插入间隔，保证各字段是对齐的

![image](https://user-images.githubusercontent.com/56336922/189623084-29d63f04-19ab-491f-b80b-dce2db133b92.png)

上图展示了x64下数据对齐的例子

### 结构体对齐例子
![image](https://user-images.githubusercontent.com/56336922/189623405-1b3ea9d8-e360-4374-b779-ea2c4e1a8151.png)

- 结构体内字段满足对齐要求
- 结构体的分布也要满足对齐要求
  - 结构体按K字节对齐，其中K为结构体内各字段最大的对齐属性
  - 初始地址和结构体长度必须是K的倍数，即初始和结束地址都为K的倍数

![image](https://user-images.githubusercontent.com/56336922/189624327-808a0684-54df-4c34-a07c-c81af284ff6e.png)

- 对于结构体数组，初始和结束地址都为K的倍数
- 结构体内元素也满足对齐要求





 

