# pypto.bitwise\_or

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

逐元素地将input与other值进行按位或（OR）运算。计算公式如下：

$$
res_i = input_i | other_i
$$

## 函数原型

```python
bitwise_or(input: Tensor, other: Union[Tensor, int]) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_INT16，DT_UINT16。 <br> 不支持空Tensor；Shape仅支持2-4维，支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为int以及Tensor类型。 <br> Tensor支持的数据类型为：DT_INT16，DT_UINT16。 <br> 不支持空Tensor；Shape仅支持2-4维，支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input、other相同，Shape为input和other广播后大小。

## 约束说明

1.  input 和 other 类型应该相同。
2.  other 为数字的时候，不支持隐式转化。
3.  other 不支持nan、inf等特殊值

## TileShape设置示例

TileShape维度应和输出一致。

如非广播场景，输入intput shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

广播场景，输入intput shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## 调用示例

```python
a = pypto.tensor([2], pypto.DT_INT16)
b = pypto.tensor([2], pypto.DT_INT16)
out = pypto.bitwise_or(a, b)
```

结果示例如下：

```python
输入数据a:  [2, 5]
输入数据b: [1, 7]
输出数据out: [3, 7]
```

