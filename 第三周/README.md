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
使用下面的等价形式将集合操作转化为为操作：
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
