# pypto.expand\_exp\_dif

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

计算自然对数的底数e的(input - other)次幂，其中other的尾轴或次尾轴的值为1，返回与输入input数据类型及形状相同的Tensor。

$$
e^(input - other)
$$

## 函数原型

```python
expand_exp_dif(input: Tensor, other: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维，并支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维，并支持按照单个维度广播到相同形状；尾轴或次尾轴的值必须为1；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，其数据类型及形状和input相同。

## 约束说明

1. input 和 other 类型应该相同。
2. other 的尾轴或次尾轴的值为1。

## TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如非广播场景，输入intput shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1, n1分别用于切分m, n轴。

广播场景，输入intput shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

## 调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.expand_exp_dif(x, y)
```

结果示例如下：

```python
输入数据x:     [[1, 2, 3], [4, 5, 6]]
输入数据y:     [[1, 2, 3]]
输出数据out:   [[ 1.      ,  1.      ,  1.      ],
               [20.085537, 20.085537, 20.085537]]
```

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([2, 1], pypto.DT_FP32)
out = pypto.expand_exp_dif(x, y)
```

结果示例如下：

```python
输入数据x:     [[1, 2, 3], [4, 5, 6]]
输入数据y:     [[1], [2]]
输出数据out:   [[ 1.       ,  2.718282 ,  7.3890557],
               [ 7.3890557, 20.085537 , 54.59815  ]]
```

