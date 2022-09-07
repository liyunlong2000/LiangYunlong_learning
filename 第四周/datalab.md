# datalab
## bitXor
### 函数要求
```
/* 
 * bitXor - x^y using only ~ and & 
 *   Example: bitXor(4, 5) = 1
 *   Legal ops: ~ &
 *   Max ops: 14
 *   Rating: 1
 */
```
函数要求实现x,y的异或操作
### 函数实现
```
int bitXor(int x, int y) {
  int res=~(~(~x&y)&~(~y&x));
  return res;
}
```
将x、y的32比特看做集合，比特为1则表示相应序号的元素在集合中，0则表示不在集合中。这样x异或y等价于集合中的x与y的对称差集。即
```
(x-y)⋃(y-x)
```
使用下面的等价形式将集合操作转化为位操作：
- 差集：x-y=~x⋂y，y-x=~y⋂x
- 交集：A⋂B=~(~A⋃B)
## tmin
### 函数要求
```
/* 
 * tmin - return minimum two's complement integer 
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 4
 *   Rating: 1
 */
```
 返回二进制补码的最小值
 ### 函数实现
 ```
 int tmin(void) {
  return 1<<31;
}
 ```
 二进制补码的最小值tmin=0x80 00 00 00，即1000 0000 0000 0000 0000 0000 0000 0000，将1左移31位即可。
 ## isTmax
 ### 函数要求
 ```
 /*
 * isTmax - returns 1 if x is the maximum, two's complement number,
 *     and 0 otherwise 
 *   Legal ops: ! ~ & ^ | +
 *   Max ops: 10
 *   Rating: 1
 */
 ```
 返回1当x是二进制补码的最大值tmax，其它情况返回0。
 ### 函数实现
 ```
 int isTmax(int x) {
  int res=(x+1)^(~(x+1)+1);
  //排除x=-1,因为x+1=0的话也是相反数对于自己
  int isNegativeOne=!(~x);
  return !(res+isNegativeOne);
}
```
二进制补码中，只有Tmin和0的相反数等于本身。当x为tmax时，x+1=tmin，只需找到相反数等于自己且不是-1的x即可。res表示(x+1)与其相反数是否相等，相等则为0。
isNegativeOne表示x是否等于-1，等于-1则返回1，不等于-1返回0。只有isNegativeOne等于0,并且res也等于0时，x才是tmax

下面为函数实现提示：
- 集合相等：使用异或^判断
- 相反数：-x=~(x)+1
## allOddBits
### 函数要求
```
/* 
 * allOddBits - return 1 if all odd-numbered bits in word set to 1
 *   where bits are numbered from 0 (least significant) to 31 (most significant)
 *   Examples allOddBits(0xFFFFFFFD) = 0, allOddBits(0xAAAAAAAA) = 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 2
 */
```
当x的奇数位都是1时，返回1。
### 函数实现
```
int allOddBits(int x) {
  int mask=0xAA;
  mask=(mask<<8)+0xAA;
  mask=mask+(mask<<16);
  int res=(x&mask)^mask;
  return !res;
}
```
找到奇数位都是1的mask(即0xAAAAAAAA)，然后用mask取出x的所有奇数位，判断是否与mask相同，相同则返回1.
下面为函数实现提示：
- 两数相等：使用异或^判断
- 掩码操作：使用&
## negate
### 函数要求
```
/* 
 * negate - return -x 
 *   Example: negate(1) = -1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 5
 *   Rating: 2
 */
```
返回x的相反数
### 函数实现
```
int negate(int x) {
  int res=~(x)+1;
  return res;
}
```
二进制补码中，取x相反数的规则是按位取反、加1.
## isAsciiDigit
### 函数要求
```
/* 
 * isAsciiDigit - return 1 if 0x30 <= x <= 0x39 (ASCII codes for characters '0' to '9')
 *   Example: isAsciiDigit(0x35) = 1.
 *            isAsciiDigit(0x3a) = 0.
 *            isAsciiDigit(0x05) = 0.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 15
 *   Rating: 3
 */
 ```
 判断x是否在[0x30,0x39]中
 ### 函数实现
 ```
 int isAsciiDigit(int x) {
  int i1=x+0x06;
  int test=0x30;
  int mask=(0xF0<<24)>>24;
  int res=(i1&mask)^test;
  //取后四位判断是否进位，不进位则为0。
  int isCarry=!!(((x&0x0f)+0x06)&0xf0);
  return !(res+isCarry);
}
```
分两部分判断：
1. 取x前28位，判断是否为0x00 00 00 3
2. 取x后4位，判断其加上6(0x06)是否产生进位

若x前28位是0x00 00 00 3，后4位加上6(0x06)不产生进位，则x是符合条件的数，返回1.
## conditional
### 函数要求
```
/* 
 * conditional - same as x ? y : z 
 *   Example: conditional(2,4,5) = 4
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 16
 *   Rating: 3
 */
```
实现条件表达式
### 函数实现
```
int conditional(int x, int y, int z) {
  int mask=((!!x)<<31)>>31;
  int res=(mask&y)+(~mask&z);
  return res;
}
```
设置mask，当x为0时mask为0，当x不为0时mask为-1.这样当x不为0时，用mask与上y，结果为y，~mask与上z，结果为0.当x为0时，用mask与上y，结果为0，~mask与上z，结果为z.
## isLessOrEqual
### 函数要求
```
/* 
 * isLessOrEqual - if x <= y  then return 1, else return 0 
 *   Example: isLessOrEqual(4,5) = 1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 24
 *   Rating: 3
 */
```
实现小于等于函数。
### 函数实现
```
int isLessOrEqual(int x, int y) {
  int sub1=y+(~x+1);
  int sub2=sub1>>31;
  int res1=!sub2;

  int res2=!(y>>31);
  int flag=!((x^y)>>31);

  //条件判断
  int mask=((!!flag)<<31)>>31;
  int res=(mask&res1)+(~mask&res2);
  return res;
}
```
考虑y-x是否大于等于0，分两部分计算：
1. x，y同号:判断y-x是否大于等于0，通过取y-x的符号位实现。
2. x，y异号：判断y是否大于等于0，通过取y的符号位实现。
然后使用条件表达式返回不同情况的结果。
## logicalNeg
### 函数要求
```
/* 
 * logicalNeg - implement the ! operator, using all of 
 *              the legal operators except !
 *   Examples: logicalNeg(3) = 0, logicalNeg(0) = 1
 *   Legal ops: ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 4 
 */
```
实现!x
### 函数实现
```
int logicalNeg(int x) {
  int i1=x|(~x+1);
  int flag=~(i1>>31);
  int res=flag&1;
  return res;
}
```
注意到只有0的或上其相反数为0，然后根据其符号位判断x是否为0，为0则返回1，不为0返回1.
## howManyBits
### 函数要求
```
/* howManyBits - return the minimum number of bits required to represent x in
 *             two's complement
 *  Examples: howManyBits(12) = 5
 *            howManyBits(298) = 10   
 *             howManyBits(-5) = 4
 *            howManyBits(0)  = 1
 * 
 *            howManyBits(-1) = 1
 *            howManyBits(0x80000000) = 32
 *  Legal ops: ! ~ & ^ | + << >>
 *  Max ops: 90
 *  Rating: 4
 */
```
计算最少的能够表示x的二进制补码位数。
### 函数实现
```
nt howManyBits(int x) {

  int sign=x>>1;
  x=(sign&~x)|(~sign&x);

  int c5,c4,c3,c2,c1,c0;
  c5=(!!(x>>16))<<4;
  x=x>>c5;

  c4=(!!(x>>8))<<3;
  x=x>>c4;

  c3=(!!(x>>4))<<2;
  x=x>>c3;

  c2=(!!(x>>2))<<1;
  x=x>>c2;

  c1=(!!(x>>1));
  x=x>>c0;

  c0=!!(x);
  int res=c5+c4+c3+c2+c1+c0+1;
  return res;
}
```
注意到，对于正数来说，结果为最高位1的序号+1，对于负数来说，可将其转为相反数，用正数的计算规则计算。采用二分法计算最高位1的序号，先将x右移16位，判断高位是否0，不为0说明1的序号大于等于16，令c5为16，为0说明序号小于16，令c5为0，最后在将x右移c5位.接着依次重复上述步骤，右移8/4/2/1位。最终结果为c5+c4+c3+c2+c1+c0+1。
## floatScale2
### 函数要求
```
/* 
 * floatScale2 - Return bit-level equivalent of expression 2*f for
 *   floating point argument f.
 *   Both the argument and result are passed as unsigned int's, but
 *   they are to be interpreted as the bit-level representation of
 *   single-precision floating point values.
 *   When argument is NaN, return argument
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
```
实现浮点数乘2.浮点数以无符号数的形式传递，符合IEEE 单精度浮点数的标准.
### 函数实现
```
unsigned floatScale2(unsigned uf) {

  //NaN
  unsigned mask1=0xff<<23;
  unsigned mask2=0x7fffff;
  unsigned sign=uf&(1<<31);
  unsigned exp=uf&mask1;
  unsigned frac=uf&mask2;
  unsigned res=0;
  unsigned add=1<<23;

  //Nan或infity为自己
  if(exp==mask1){
    res=uf;
    return res;
  }
  //0和非规格化小数则乘2加上符号位
  if(exp==0){
    res=sign|(uf<<1);
    return res;
  }
  uf+=add;
  if((uf&mask1)==mask1){
    res=sign|mask1;
    return res;
  }
  return uf;
}
```
分别取出浮点数的符号位sign、指数位exp、尾数位frac，分三种情况进行处理:
1. 当浮点数uf为Nan或infity(即exp位全1),乘以2后结果为自己。
2. 当浮点数uf为0或非标准化小数时(即exp位全0)，结果为uf左移一位并或上符号位，不会发生溢出。
3. 其余情况，浮点数uf的exp域加上1即可，当exp全1时，返回infity或上符号位，否则，直接返回。
## floatFloat2Int
### 函数要求
```
/* 
 * floatFloat2Int - Return bit-level equivalent of expression (int) f
 *   for floating point argument f.
 *   Argument is passed as unsigned int, but
 *   it is to be interpreted as the bit-level representation of a
 *   single-precision floating point value.
 *   Anything out of range (including NaN and infinity) should return
 *   0x80000000u.
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
```
实现浮点数强制转为整型，浮点数以无符号数的形式传递，符合IEEE 单精度浮点数的标准.
### 函数实现
```
int floatFloat2Int(unsigned uf) {
  int mask1=0xff<<23;
  int mask2=0x7fffff;
  int sign=uf&(1<<31);
  int exp=(uf&mask1)>>23;
  int E=exp-127;
  int frac=uf&mask2;
  int res,M;
  if(E<0){
    res=0;
  }else if(E>31){
    res=0x80000000;
  }else{
     M=0x800000+frac;
     if(E<=23){
      res=M>>(23-E);
     }else{
      res=M<<(E-23);
     }
     if(sign){
      res=(~res)+1;
    }
  }
  return res;
}
```
首先取出浮点数的实际指数E=exp-bias，其中bias=127.然后根据E分情况讨论：
1. 当E小于0时，尾数部分为xxxx，则实际小数为1.xxxx或0.xxxx。经过右移后，为0.
2. 当E大于31时，尾数部分为xxxx，则实际小数为1.xxxx，经过左移后，高位1溢出，返回溢出值0x80000000。
3. 当E$le$23时，尾数部分为xxxx，则实际小数为1.xxxx，尾数后面需要截断E-23个0，即frac先加上隐藏的1，然后右移23-E位。
4. 当E>23时，尾数部分为xxxx，则实际小数为1.xxxx，尾数后面需要补E-23个0，即frac先加上隐藏的1，然后左移E-23位。

## floatPower2
### 函数要求
```
/* 
 * floatPower2 - Return bit-level equivalent of the expression 2.0^x
 *   (2.0 raised to the power x) for any 32-bit integer x.
 *
 *   The unsigned value that is returned should have the identical bit
 *   representation as the single-precision floating-point number 2.0^x.
 *   If the result is too small to be represented as a denorm, return
 *   0. If too large, return +INF.
 * 
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. Also if, while 
 *   Max ops: 30 
 *   Rating: 4
 */
```
实现$2.0^x$,返回以无符号数表示的，满足单精度浮点类型的浮点数结果。当结果过小时，返回0，当结果过大时，返回+infity。
### 函数实现
```
unsigned floatPower2(int x) {
  int res,exp;
  if(x>=128) 
    res=0x7f800000;
  else if(x<=-127)
    res=0;
  else{
    exp=x+127;
    exp=exp<<23;
    res=exp;
  }
  return res;
}
```
标准化的单精度小数的指数范围为[-127,128],根据传入的参数x分情况讨论：
1. 当x$\geq$128时,结果发生溢出，返回溢出值0x7f800000。
2. 当x$\leq$-127时，结果为非规格化小数或极小的无法表示的数，返回0.
3. 其余情况，为规格化小数，计算出exp域值，然后左移23位即为结果。