# pypto.Element 构造函数

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

创建Element

## 函数原型

```python
def __init__(self, dtype, data) : ...
```

## 参数说明


| 参数名 | 输入/输出 | 说明                  |
|--------|-----------|-----------------------|
| dtype  | 输入      | 数据类型，详见<a href="../datatype/DataType.md">DataType</a> |
| value  | 输入      | 整数或者浮点数        |

## 返回值说明

返回Element

## 约束说明

无。

## 调用示例

```python
t = pypto.Element(pypto.DT_FP32, 3)
```

