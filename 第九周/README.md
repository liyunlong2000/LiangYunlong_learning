# C++智能指针
- 作用：主要在栈上创建的对象，用来管理堆上的对象。智能指针能够更好地管理堆内存，使用普通指针，容易造成堆内存泄漏（忘记释放）、二次释放、程序执行异常时的内存泄漏等问题。
- 实现原理：通用实现技术是使用引用计数(reference count)来管理共享的对象，当引用计数值减为0时，即可释放对象资源
## auto_ptr指针
- 在头文件`memory`中定义：`template <class X> class auto_ptr;`
- `auto_ptr`对象具有对分配给它们的指针拥有所有权的特性：对一个元素拥有所有权的`auto_ptr`对象负责销毁它指向的元素，并在销毁它时销毁分配给它的内存。析构函数通过调用算子`delete`来自动执行此操作。
- 不会有两个`auto_ptr`对象拥有相同的元素，当两个`auto_ptr`对象之间发生复制操作时，所有权转移，失去所有权的`auto_ptr`对象被设置为空指针

### 初始化
```
//通过构造函数
auto_ptr<Test> ptest(new Test("123"));
//通过拷贝构造函数
auto_ptr<Test> ptest(ptest2);
//通过reset()绑定对象
auto_ptr<Test> ptest();
Test* t1=new Test("123");
ptest.reset(t1);
```
## unique_ptr指针
- 在头文件`memory`中定义：`template <class X> class unique_ptr;`
- `unique_ptr`是取代`auto_ptr`的产物
  - `auto_ptr`中复制操作容易产生空指针引用
  - `auto_ptr`中使用非数组的`delete`，容易导致内存泄漏
### 初始化
```
//通过构造函数
unique_ptr<Test> ptest(new Test("123"));
//通过移动赋值
unique_ptr<Test> ptest2();
ptest2 = std::move(ptest); 
//通过reset()绑定对象
auto_ptr<Test> ptest();
Test* t1=new Test("123");
ptest.reset(t1);
//通过make_unique
auto_ptr<Test> ptest(make_unique<Test>("12345"));
```
### 值传递
```
void func(unique_ptr<Test> ptest){
    ptest->setStr("Hello");
    ptest->print();
    return;
} 
```
- `unique_ptr`对象无法直接值传递给函数,因为默认的拷贝构造和拷贝赋值函数为`delete`
- 可以使用移动语义将`unique_ptr`对象的所有权转移到函数形参中。此时，原本`unique_ptr`对象指针赋值为空.
```
func(move(ptest1));
```
### 引用传递
```
void func2(unique_ptr<Test> &ptest){
    ptest->setStr("Hello");
    ptest->print();
    return;
}
```
- 可以直接将`unique_ptr`对象传递给函数
## shared_ptr指针
- 在头文件`memory`中定义:`template< class T > class shared_ptr;`
- 多个`shared_ptr`可以管理同一对象，它们之间共享一个引用计数
  - 当拷贝一个`shared_ptr`时,引用计数加1
  - `shared_ptr`中可以使用`use_count()`返回引用计数值
  - 当`shared_ptr`共享的引用计数降为0时,所管理的对象自动被析构(调用其析构函数)

- 风险:如果管理资源的任何`std :: shared_ptr`未被正确销毁，则资源将无法正确释放。

## weak_ptr指针
- 在头文件`memory`中定义:`template< class T > class weak_ptr;`
- 用来解决`shared_ptr`循环引用产生的死锁问题
  - 若两个指针相互引用形成了环,那么引用计数不可能为0.资源得不到释放
  - `weak_ptr`是对共享对象的一种弱引用，不会增加对象的引用计数
- 通过`share_ptr`对`weak_ptr`进行拷贝构造,初始化
- 使用`lock()`函数得到`weak_ptr`对象所管理的`shared_ptr`对象

```
void fun()
{
    shared_ptr<B<A>> pb(new B<A>());
    shared_ptr<A> pa(new A());
    pb->pa_ = pa;
    pa->pb_ = pb;
    cout<<"fun : "<<pb.use_count()<<endl;
    cout<<"fun : "<<pa.use_count()<<endl;
    // 这个函数执行完会出现相互引用导致的内存泄漏
}
```
- `pb`中`pa_`属性类型为`shared_ptr<A>`
- `pa`中`pb_`属性类型为`shared_ptr<B<A>>`
- `fun`函数中`pa`和`pb`之间相互引用，引用计数为2.当跳出函数时，`pa`和`pb`超出作用域，它们析构时使得引用计数减一，得到1.`std :: shared_ptr`未被正确销毁，则资源`A`和`B`将无法正确释放。
```
void fun1()
{
    shared_ptr<B<A1>> pb(new B<A1>());
    shared_ptr<A1> pa(new A1());
    pb->pa_ = pa;
    pa->pb_ = pb;
    cout<<"fun1 : "<<pb.use_count()<<endl;
    cout<<"fun1 : "<<pa.use_count()<<endl;
}
```
- `pb`中`pa_`属性类型为`shared_ptr<A1>`
- `pa`中`pb_`属性类型为`weak_ptr<B<A>>`
- `fun`函数中`pa`和`pb`之间相互引用，`pa`引用计数为2,`pb`引用计数为1.当跳出函数时，`pa`和`pb`超出作用域，它们析构时使得引用计数减一，pb引用计数为0,资源`B`被释放.这样资源`A`的引用计数也减一(B中`pa_`被释放)。
- `pa`析构时计数为0，资源`A`被释放

