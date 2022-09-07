# datalab
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
3. 当0 $\leq$ E $\leq$ 23时，尾数部分为xxxx，则实际小数为1.xxxx，尾数后面需要截断E-23个0，即frac先加上隐藏的1，然后右移23-E位。
4. 当23< E $\leq$ 31 时，尾数部分为xxxx，则实际小数为1.xxxx，尾数后面需要补E-23个0，即frac先加上隐藏的1，然后左移E-23位。

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
实现 $2.0^x$ ,返回以无符号数表示的，满足单精度浮点类型的浮点数结果。当结果过小时，返回0，当结果过大时，返回+infity。
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
1. 当x $\geq$ 128时,结果发生溢出，返回溢出值0x7f800000。
2. 当x $\leq$ -127时，结果为非规格化小数或极小的无法表示的数，返回0.
3. 其余情况，为规格化小数，计算出exp域值，然后左移23位即为结果。
