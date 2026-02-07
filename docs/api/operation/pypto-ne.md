# pypto.ne

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

逐元素不等于比较运算。

## 函数原型

```python
ne(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16, DT_BF16, DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为：Tensor, float, Element。 <br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16, DT_BF16, DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Shape与输入Tensor一致、数据类型为DT\_BOOL的Tensor。若input对应位置的元素值不等于other对应位置的元素值，则该位置的返回值为True，其余位置的返回值为False。

## 约束说明

1.  input 和 other 类型须保持一致。
2.  支持一维广播。

## TileShape设置示例

TileShape维度应和输出一致。

如非广播场景，输入intput shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

广播场景，输入intput shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## 调用示例

```python
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.ne(a, b)
```

结果示例如下：

```python
输入数据a: [1.0 2.0 3.0]
输入数据b: [2.0 2.0 2.0]
输出数据out: [True, False, True]
```

