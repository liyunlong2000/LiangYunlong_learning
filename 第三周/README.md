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
