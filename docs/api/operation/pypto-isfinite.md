# pypto.isfinite

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

判断张量中的元素值是否为有限值。

当张量类型为整型时，返回一个全为 True 的布尔类型、与输入张量形状大小一致的张量。

当张量类型为浮点数类型时，仅 inf/nan/-inf 不是有限值，结果中对应元素位置的值为 False，其余为 True。

## 函数原型

```python
isfinite(self: Tensor) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| self   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32，DT_UINT8，DT_INT8，DT_UINT16，DT_INT16，DT_UINT32，DT_INT32，DT_UINT64，DT_INT64。 <br> 不支持空Tensor；形状大小支持 1-5 维；形状大小中对应元素的个数不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出张量，张量的数据类型为布尔类型 DT_BOOL，形状大小与输入张量的形状大小一致。

## 约束说明

1. 仅支持 DT_FP16，DT_BF16，DT_FP32，DT_UINT8，DT_INT8，DT_UINT16，DT_INT16，DT_UINT32，DT_INT32，DT_UINT64，DT_INT64 等数据类型。
2. TileShape 以及 ViewShape 的尾轴必须按照输出张量的类型 32B 对齐，由于输出张量为布尔类型，因此 TileShape 以及 ViewShape 的尾轴必须是 32 的倍数。

## TileShape设置示例

TileShape 维度应和输出一致。

假设，输入形状为 [m, n]，输出为 [m, n]，TileShape 设置为 [m1, n1], 则 m1, n1 分别用于切分 m, n 轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## 调用示例

```python
self = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
out = pypto.isfinite(self)
```

结果示例如下：

```python
输入数据 self: [[1 nan 3],
               [inf 1 1],
               [1, 1, -inf]]
输出数据 out: [[True False True],
             [False True True],
             [True True False]]
```

