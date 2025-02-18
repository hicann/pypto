# pypto.SymbolicScalar.is\_immediate

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

判断符号标量是否为立即值（具体数值）。

## 函数原型

```python
is_immediate(self) -> bool
```

## 参数说明

无

## 返回值说明

如果是立即值返回True，不是返回False。

## 约束说明

立即值是指在编译时就能确定具体数值的常量

## 调用示例

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_immediate()
out2 = s2.is_immediate()
```

结果示例如下：

```python
输出数据out1: True
输出数据out2: False
```

