# pypto.cbrt

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

计算输入Tensor中每个元素的立方根，逐元素运算，返回与输入形状相同的Tensor。计算公式为：

$$
y_i = \sqrt[3]{x_i}
$$

## 函数原型

```python
cbrt(input: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16， DT_BF16。 <br> 不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape与input相同。

## 约束说明

1. 输入Tensor和输出Tensor类型应该相同。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入input shape为[m, n]，输出为[m, n], TileShape设置为[m1, n1],则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([1.0, 2.0], pypto.DT_FP32)
y = pypto.cbrt(x)
```

结果示例如下：

```python
Input  x:[[8.0, -8.0]]
Output y:[[2.0, -2.0]]
```
