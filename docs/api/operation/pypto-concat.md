# pypto.concat

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

将输入的多个Tensor沿指定维度（dim）拼接，返回一个拼接后的Tensor。

## 函数原型

```python
concat(tensors: List[Tensor], dim: int = 0) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| tensors | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_BF16, DT_FP32, DT_FP16, DT_INT8, DT_INT16, DT_INT32。 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 源操作数。 <br> 支持的数据类型为：int，默认为0。                      |

## 返回值说明

返回输出Tensor，Tensor的数据类型和 tensors 的任一tensor数据类型相同，Shape与 tensors任一tensor相同（除 dim 对应维度），dim 对应维度为 tensors 各个 tensor 对应维度之和。

## 约束说明

1.源操作数 tensors 的大小需要大于等于 2，即 len\(tensors \)\>=2；小于等于128。（支持输入一个tensor情况，精度暂时不保证）；

2.输入 tensor 数据类型相同、维度数量相同，并且除待拼接维度（dim）之外的每个维度值相同；

3.dim: -input.dim <= dim < input.dim（input对应tensors 的任一tensor）；

4.设置viewshape时，dim对应维度不切块（即viewshape对应值\>=tensors 任一tensor的对应值）。

## TileShape设置示例

TileShape维度应和输出一致。

如输入tensors维度为[m, c1, p], [m, c2, p]，输出为[m, c1+c2, p], TileShape设置为[m1, n1, p1], 则m1, p1分别用于切分m, p轴, n1用于切分c1和c2轴。

```python
pypto.set_vec_tile_shapes(m1, n1, p1)
```

## 调用示例

```python
a = pypto.tensor([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 1s
b = pypto.tensor([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 0s
out = pypto.concat([a, b], dim = 0) 
```

结果示例如下：

```python
输入数据a:   [[1.0 1.0],
              [1.0 1.0]]
输入数据b:   [[0.0 0.0],
              [0.0 0.0]]
输出数据out: [[1.0 1.0],
              [1.0 1.0],
              [0.0 0.0],
              [0.0 0.0]]

```

