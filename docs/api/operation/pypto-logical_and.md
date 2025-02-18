# pypto.logical\_and

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

对两个输入的Tensor行逐元素逻辑与（AND）运算。运算规则：

-   如果输入的Tensor为 bool 则 True and True -\> True，其余情况皆为 False。
-   如果输入的Tensor数值，会自动转换成 True/False，0 为 False，非 0 为 True。

## 函数原型

```python
logical_and(input: Tensor, other: Tensor) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16, INT8, UINT8, BOOL。 <br> 不支持空Tensor；Shape仅支持2-4维，支持输入Tensor的数据类型不同，支持广播。Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16, INT8, UINT8, BOOL。 <br> 不支持空Tensor；Shape仅支持2-4维，支持输入Tensor的数据类型不同，支持广播。Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型为DT\_BOOL，形状为广播后的形状。

## 约束说明

1.  TileShape与input、other维度保持一致；
2.  由于存在临时内存使用，TileShape大小有额外约束，假设TileShape为\[a,b,c,d\]，那么a\*b\*c\*d\*sizeof\(self\) + a\*b\*c\*d\*sizeof\(other\) + a\*b\*c\*d\*sizeof\(BOOL\) + 1.1875KB<UB。

## 调用示例

```python
x = pypto.tensor([2], pypto.DT_BOOL)
y1 = pypto.tensor([2], pypto.DT_BOOL)
z1 = pypto.logical_and(x, y1)
# 支持广播
y2 = pypto.tensor([2,2], pypto.DT_BOOL)
z2 = pypto.logical_and(x, y2)
```

结果示例如下：

```python
输入数据x:  [True, False]
输入数据y1: [True, True]
输入数据y2: [[True, False], [False, True]]
输出数据z1: [True, False]
输出数据z2: [[True, False], [False, False]]
```

