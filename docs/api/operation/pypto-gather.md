# pypto.gather

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

对输入的 input，按照指定维度 dim 和索引 index 提取原始 Tensor的对应值，最后返回结果。例如对3维 Tensor，有以下计算公式：

$$
\begin{cases}
output[i,j,k] = input[index[i,j,k], j, k] & \text{if } dim = 0; \\
output[i,j,k] = input[i, index[i,j,k], k] & \text{if } dim = 1; \\
output[i,j,k] = input[i,j, index[i,j,k]] & \text{if } dim = 2.
\end{cases}
$$

## 函数原型

```python
gather(input: Tensor, dim: int, index: Tensor) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT32。 <br> 不支持空 Tensor，Shape 支持2-4维，且shape size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 源操作数。 <br> 支持任意合法的维度索引 ，范围为：-input.dim 到 input.dim - 1。 |
| index   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_INT32, DT_INT64。 <br> 不支持空 Tensor，Shape 支持2-4维，需保证 index 所有轴上的 Shape 大小不超过 input 的对应 Shape 大小，且值为合法索引，即不超过 input 在 dim 轴上的 Shape 大小。 |

## 返回值说明

返回输出 Tensor，输出 Tensor数据类型与 input 数据类型保持一致；输出 Tensor的Shape 与 index 的 Shape 相同。

## 约束说明

1. index.dim = input.dim，且 index.shape\[i\] <= input.shape\[i\] (i != dim)，值为合法索引，即不能超出 input.shape\[dim\]；

2. dim: -input.dim <= dim < input.dim；

3. input.shape 的 dim 轴不可切，要求 viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \)，其余维度的 Shape 大小不做限制；

4. TileShape的维度与 result 相同，用于切分 result 和 index，TileShape\[dim\] = viewshape\[dim\]，所有输入和输出的 TileShape 大小总和不能超过UB内存的大小。

## TileShape设置示例

TileShape维度应和输出一致。

如输入input为[x, y, z], dim为1，输入index为[m, t, p]，输出为[m, t, p], 其中 m <= x, p <= z, TileShape设置为[m1, t1, p1], 则m1, t1, p1分别用于切分m, t, p轴。 y轴不可切，必须保证y轴全载。

```python
pypto.set_vec_tile_shapes(m1, t1, p1)
```

## 调用示例

```python
x = pypto.tensor([3, 5], pypto.DT_INT32)        # shape (3, 5)
index = pypto.tensor([3, 4], pypto.DT_INT32)   # shape (3, 4)
dim = 0
y = pypto.gather(x, dim, index)
```

结果示例如下：

```python
输入数据 x: [[0,  1,  2,  3,  4],
             [5,  6,  7,  8,  9],
             [10, 11, 12, 13, 14]]
     index: [[0, 1, 2, 0],
             [1, 2, 0, 1],
             [2, 2, 1, 0]]
输出数据 y: [[0,  6,  12, 3],
             [5,  11, 2,  8],
             [10, 11, 7,  3]]
```

