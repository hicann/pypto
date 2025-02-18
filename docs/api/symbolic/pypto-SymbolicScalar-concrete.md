# pypto.SymbolicScalar.concrete

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取符号标量的具体数值。

## 函数原型

```python
concrete(self) -> int
```

## 参数说明

无

## 返回值说明

符号标量的具体数值。

## 约束说明

-   只有在 is\_concrete\(\) 返回 True 时才能调用此方法
-   如果符号标量不是具体的，将抛出 ValueError 异常

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.concrete()

# 如果符号标量没有具体值会抛出异常
s2 = pypto.SymbolicScalar("x")
out2 = s2.concrete()
```

结果示例如下：

```python
输出数据out1: 10
输出数据out2: 抛出异常 ValueError: Not concrete value
```

