# pypto.SymbolicScalar.min

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

计算两个符号标量的最小值。

## 函数原型

```python
min(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar'
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | 输入      | 待比较的符号标量或整数。 |

## 返回值说明

两个值中的最小值。

## 约束说明

-   如果两个值都是具体的，将返回具体的常量值
-   如果至少有一个不是具体的，将返回符号表达式

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar(5)
out1 = s1.min(s2)
out2 = s1.min(3)
s3 = pypto.SymbolicScalar(x)
out3 = s3.min(2)
```

结果示例如下：

```python
输出数据out1: 5
输出数据out2: 3
输出数据out3: RUNTIME_Min(x, 2)
```

