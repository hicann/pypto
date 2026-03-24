# pypto.remainder

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

将 input 的每个元素和 other 中对应位置的元素进行取余运算，计算公式如下：

$$
res_i = input_i - other_i * floor(input_i / other_i)
$$

## 函数原型

```python
remainder(input: Union[Tensor, int, float], other: Union[Tensor, int, float]) -> Tensor
```

## 参数说明


| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor、int、float。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16。 <br> 不支持空Tensor；Shape仅支持1-5维，并支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。 <br> 支持的类型为：Tensor、int、float。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16。 <br> 不支持空Tensor；Shape仅支持1-5维，并支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Shape为input和other广播后大小，Tensor的数据类型和input、other相同。

## 约束说明

1. input 和 other 类型相同；
2. other 不支持0等特殊值；
3. int32在数据范围超过\[-2^24, 2^24\]时不保证精度。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如非广播场景，输入intput shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1, n1分别用于切分m, n轴。

广播场景，输入intput shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP32)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP32)
out = pypto.remainder(a, b)
```

结果示例如下：

```python
输入数据a:    [7.0, 8.0, 9.0] 
输入数据b:    [-3.0, -3.0, -3.0]
输出数据out:  [-2.0, -1.0, 0.0]
```