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
## 代码中QKV-fuse
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

# CMake
## 简介
CMake的目的是根据`CMakeLists.txt`生成可移植的Makefile，使用者可以方便地构建工程。
## 项目运行
在`CMakeLists.txt`所在目录中执行以下命令
```
mkdir build
cd build
cmake ..
make
```
生成目标文件或可执行文件。

## 描述性命令
- `cmake_minimum_required(VERSION 3.15)`：指定了CMake的最小版本，一些高版本命令无法在低版本中使用
- `project(Tutorial VERSION 1.0)`：指定了项目的名称和版本号
  - `PROJECT_NAME`：变量将会被赋值，本例中即Tutorial，可以使用`${PROJECT_NAME}`使用该值
  - `PROJECT_SOURCE_DIR`：当前工程的源码路径
  - `PROJECT_BINARY_DIR`：当前工程的二进制路径——CMakeCache.txt所在的目录，一般情况即build目录
  
## 指定C++标准
通过以下两个命令指定C++的标准：
```
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```

## add_library添加库
```
add_library(<name> [STATIC | SHARED | MODULE]
            [EXCLUDE_FROM_ALL]
            [<source>...])
```
根据源码生成一个库
## add_subdirectory添加子目录
```
add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
```
使用该命令可以将子目录加入构建系统，这样就可以利用子目录中生成的库。
## add_executable
```
add_executable(<name> [WIN32] [MACOSX_BUNDLE]
               [EXCLUDE_FROM_ALL]
               [source1] [source2 ...])
```
使用该命令根据指定的源文件生成可执行文件。
## target_link_libraries链接库
```
target_link_libraries(<target>
                      <PRIVATE|PUBLIC|INTERFACE> <item>...
                     [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
```
使用该命令为目标文件链接所依赖的库。
## target_include_directories头文件搜索路径
```
target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
  <INTERFACE|PUBLIC|PRIVATE> [items1...]
  [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
```
使用该命令将指定目录加入到include头文件搜索路径中。

## PUBLIC、PRIVATE、INTERFACE关键字
### 含义
上面三个关键字用来控制依赖或传播，例如`target_link_libraries(A B)`命令：
- `PRIVATE`：依赖项B仅链接到目标A，若有C链接了目标A，C不链接依赖项B
  - 若C链接目标A，但是C不使用B中实现，使用PRIVATE
- `INTERFACE`：依赖项B并不链接到目标A，若有C链接了目标A，C会链接依赖项B
  - 若C链接目标A，但是C不使用A中实现，而使用B中实现，使用INTERFACE
- `PUBLIC`：依赖项B链接到目标A，若有C链接了目标A，C也会链接依赖项B
  - 若C链接目标A，C不仅使用A中实现，也使用B中实现，使用PUBLIC
  
### 例子
```
.
├── MathFunctions
│   ├── CMakeLists.txt
│   ├── MathFunctions.h
│   └── mysqrt.cxx
├── CMakeLists.txt
├── TutorialConfig.h.in
└── tutorial.cxx
```
对于以上目录结构，子目录`MathFunctions`中`CMakeLists.txt`根据`mysqrt.cxx`生成库`MathFun`,对外的头文件声明为`MathFunctions.h`。顶级目录中`CMakeLists.txt`根据`tutorial.cxx`生成可执行文件`tutorial`，而`tutorial.cxx`需要使用库`MathFun`。

顶级目录中`CMakeLists.txt`需要包含头文件`MathFunctions.h`所在的目录，那么可在子目录中`CMakeLists.txt`加入下面命令：
```
target_include_directories(MathFun
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
```
这样，可执行文件`tutorial`编译时会include到路径`${CMAKE_CURRENT_SOURCE_DIR}`，不需要显示添加include路径。

# 编译原理PW2 用不同的方法构造词法分析器
## 第1关：手工编写一个识别关系运算符的词法分析器
### 实验说明
#### labLexer-1.cpp
![image](https://user-images.githubusercontent.com/56336922/190942929-f1554a4a-2a12-424e-9c5d-6d2230a82be8.png)

- 对输入字符流逐个读取字符c
- 若c为关系运算符，根据关系运算符的转换图对读入的关系运算符进行分析，注意带星号* 的接收状态需要使用`ungetc(c,stdin);`将读入的字符c返回标准缓冲区中
- 若c为非关系运算符，则循环读入字符c，维护计数器cnt，直到c为关系运算符或换行符

#### CMakeList-1.txt
```
add_executable(labLexer-1 ${labLexer_SOURCE_DIR}/src/labLexer-1.cpp)
```
`CMakeList-1.txt`中使用上述命令生成可执行文件`labLexer-1`即可。
### 注意事项
- `LF`："\n"，Linux的换行符
- `CRLF`："\r\n"，Windows的换行符

## 第2关：用 Flex 构造识别关系运算符的词法分析器
### 实验说明
#### relop.lex
```
%{
Declarations
%}
Definitions
%%
Rules
%%
User subroutines
```
.lex文件的结构如上：
- `Declarations`：声明内容将会直接复制到所生成`lex.yy.c`中，一般在这里声明一些全局变量和函数，这样在后面可以使用这些变量和函数
- `Definitions`：定义正则表达式中的一些名字，可以在规则（Rules）段被使用
- `Rules`：每一行为一条规则，规则由匹配模式（pattern）和事件（action）组成，模式在前面，用正则表达式表示，事件在后面，即 C 代码。每当一个模式被匹配到时，后面的 C 代码被执行。
- `User subroutines`：内容会被原样复制到 `lex.yy.c` 文件的最末尾，一般声明用户需要用到的辅助函数。

flex会将.lex文件翻译为一个`yylex()`函数，该函数扫描输入文件，当扫描到一个完整的、最长的、可以和某条规则的正则表达式所匹配的字符串时，该函数会执行此规则后面的 C 代码。如果这些 C 代码中没有 `return` 语句，则执行完这些 C 代码后， `yylex()` 函数会继续运行，开始下一轮的扫描和匹配。

`relop.lex`文件：
- 定义`relop ("<"|"<="|"<>"|"="|">"|">=")`，当匹配到`relop`时，输出其内容，内容可用变量`yytext`表示
- 定义`other (" "|{letter}|"("|")")+`,其中`letter`为`letter`，当匹配到`other`时，输出其长度，可用变量`yyleng`表示
- 当匹配到`\n`时，返回0，程序扫描结束
- `User subroutines`中定义了`getsym()`函数，是对`yylex()`函数的一层封装
#### labLexer-2.cpp
- 使用条件编译，当没有定义宏`LEXERGEN`时，则使用第一关中封装的`getsym()`函数来处理输入。当定义了宏`LEXERGEN`时，则调用`lex.yy.h`头文件中声明的`getsym()`函数来处理输入

#### Makefile
- 使用`all:labLexer-2 labLexer-2m`，生成两个可执行程序
- 使用`-I ./grammar`，添加`lex.yy.h`头文件所在目录
- 生成`labLexer-2`可执行文件时，需要链接由.lex文件生成的`lex.yy.c`，调用封装好的`getsym()`函数来处理输入

## 第3关：用 ANTLR 构造识别关系运算符的词法分析器
### 实验说明
#### relop.g4
定义需要匹配的token：
- 定义`Relop:'<'|'<='|'<>'|'='|'>'|'>=';`
- 定义`Other:('\t'|' '|[a-zA-Z]|'('|')')+;`
- 定义`End: '\n' -> skip;`

#### labLexer-3.cpp
- 读入一行字符串，调用api得到tokens
- 根据token的类型进行输出，可以使用`token->getType()`函数获得类型
  - 当`type==1`时，即为`Relop`，使用`token->getText()`函数输出匹配到token的内容
  - 当`type==2`时，即为`Other`，使用`token->getStopIndex()`和`token->getStartIndex()`得到token的长度
# 实验链接
[第1关：手工编写一个识别关系运算符的词法分析器](https://www.educoder.net/tasks/6p4fzmqfy3ba)

[第2关：用 Flex 构造识别关系运算符的词法分析器](https://www.educoder.net/tasks/goqkv2sln4xh)

[第3关：用 ANTLR 构造识别关系运算符的词法分析器](https://www.educoder.net/tasks/twc9mfy7b8ir)
