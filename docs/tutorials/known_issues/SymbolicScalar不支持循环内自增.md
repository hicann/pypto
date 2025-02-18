# SymbolicScalar不支持循环内自增

## 问题现象描述

```python
@pypto.jit
def add_kernel_1(a, b, c):
    count = 0
    for i in pypto.loop(20):
        count = count + 1
```

当实际执行到i = 1时，count并不会像用户预期的那样从0依次增加到20。

## 可能原因

当前PyPTO框架仅捕获了用户的Tensor操作，而未捕获用户的scalar操作，因此不会将count处理为变量。目前，只有循环变量能够实现自增。

## 处理步骤

使用循环变量来表达自增逻辑。

