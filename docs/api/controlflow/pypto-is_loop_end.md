# pypto.is\_loop\_end

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

定义一个循环操作，实现python当中的for循环功能。

## 函数原型

```python
def is_loop_end(scalar: SymInt) -> SymbolicScalar
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| scalar  | 输入      | 当前循环的索引。 |

## 返回值说明

返回一个符号标量表达式，表示是否为循环结束（布尔值）

## 约束说明

-   scalar 必须是循环迭代器返回的符号标量
-   如果不是循环索引，将抛出 ValueError 异常

## 调用示例

```python
for idx in pypto.loop(0, 10, 1):
    if pypto.cond(pypto.is_loop_end(idx)):
        ...
```

