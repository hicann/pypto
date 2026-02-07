# pypto.div

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

将 input 的每个元素除以 other 中对应位置的元素，计算公式如下：

$$
res_i = input_i \div other_i
$$

## 函数原型

```python
div(input: Tensor,other: Union[Tensor, float]) -> Tensor
```

## 参数说明


| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_BF16, DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维，并支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。 <br> 支持的类型为float以及Tensor类型。 <br> Tensor支持的数据类型为：DT_FP16，DT_BF16, DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维，并支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

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
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.div(a, b)
```

结果示例如下：

```python
输入数据a:    [[2.0 4.0 6.0]] 
输入数据b:    [[2.0 2.0 2.0]]
输出数据out:  [[1.0 2.0 3.0]]
```

