# pypto.pow

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

计算输入 Tensor中每个元素的 other 次幂，逐元素运算，返回与输入形状相同的Tensor。

## 函数原型

```python
pow(input: Tensor, other: Union[Tensor, int, float]) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16、DT_BF16、DT_FP32、DT_INT32。 <br> 不支持空Tensor；Shape仅支持2-4维；支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 指数。 <br> 支持的类型为Tensor、int或float。 <br> Tensor支持的数据类型为：DT_FP16、DT_BF16、DT_FP32、DT_INT32。 <br> 不支持空Tensor；Shape仅支持2-4维；支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回一个与输入形状相同、数据类型一致的Tensor，其元素为输入Tensor对应元素的other次幂。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入intput shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
a = 2
b = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.pow(x, a)
z = pypto.pow(x, b)
```

结果示例如下：

```python
输入数据x: [[1.0  2.0], [-3.0  4.0]]
输入数据b: [[2.0  2.0], [1.0   1.0]]
输出数据y: [[1.0  4.0], [9.0  16.0]]
输出数据z: [[1.0  4.0], [-3.0  4.0]]
```

