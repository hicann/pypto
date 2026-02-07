# pypto.maximum

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

计算输入与另一输入的逐元素最大值。支持二维、三维或四维的Tensor。

## 函数原型

```python
maximum(
    input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]
) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为 int, float, Element, Tensor类型。 <br> 当为 int 或者 float 类型时会自动转换为 Element 类型，其中 int 对应 DT_INT_32，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_BF16, DT_INT16，DT_INT32，DT_FP32 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为 int, float, Element, Tensor类型。 <br> 当为 int 或者 float 类型时会自动转换为 Element 类型，其中 int 对应 DT_INT_32，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_BF16, DT_INT16，DT_INT32，DT_FP32 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 <br> 类型和数据类型必须与源操作数一保持一致。 |

源操作数一与源操作数二之间至少一者为Tensor。

## 返回值说明

当两个源操作数均为Tensor时，两个Tensor必须满足广播关系。该接口返回一个与源操作数一和源操作数二广播后形状相同的Tensor，数据类型与源操作数相同，其元素为源操作数一和源操作数二的逐元素最大值。且源操作数为Tensor时，源操作数一和源操作数二均仅支持单轴广播。

当两个源操作数之中存在一个Tensor时，返回与输入Tensor相同形状的Tensor，其元素为源操作数一和源操作数二的逐元素最大值。

## TileShape设置示例

TileShape维度应和输出一致。

如非广播场景，输入intput shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

广播场景，输入intput shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## 调用示例

```python
a = pypto.tensor([3], pypto.DT_INT32)
b = pypto.tensor([3], pypto.DT_INT32)
out = pypto.maximum(a, b)
```

结果示例如下：

```python
输入数据a: [0, 2, 4]
输入数据b: [3, 1, 3]
输出数据out: [3, 2, 4]
```

