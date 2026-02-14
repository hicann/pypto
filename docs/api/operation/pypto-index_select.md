# pypto.index\_select

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

返回一个新的张量，该张量使用索引 `index` 中的元素沿维度 `dim` 对输入张量进行索引。

返回的张量与原始张量（输入）具有相同的维度数。第 `dim` 维度的大小与索引 `index` 的长度相同；其他维度的大小与原始张量相同。

$$
\begin{array}{l}
\text{shape}(\mathbf{input}) = (S_0, S_1, \ldots, S_{n-1}) \\
dim = d \\
\text{shape}(\mathbf{index}) = (I_0,) \\
\text{shape}(\mathbf{result}) = (S_0, \ldots, S_{d-1}, I_0, S_{d+1}, \ldots, S_{n-1}) \\
\mathbf{result}[s_0, \ldots, s_{d-1}, i, s_{d+1}, \ldots, s_{n-1}] = \mathbf{input}[s_0, \ldots, s_{d-1}, \mathbf{index}[i], s_{d+1}, \ldots, s_{n-1}]
\end{array}
$$
## 函数原型

```python
index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT16，DT_INT8，DT_INT32。 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | int 类型，索引的维度； <br> 支持任意不超过 input 维数的值，详见约束说明。 |
| index   | 输入      | 源操作数； <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_INT32，DT_INT64； <br> 不支持空 Tensor，Shape只支持1-2维；Shape Size不大于2147483647（即INT32_MAX），且值为合法索引，即不超过 input 在 dim 轴上的 Shape 大小。

## 返回值说明

返回输出 Tensor，输出 Tensor数据类型与 input 数据类型保持一致；输出 Tensor的Shape 有input、 dim 以及 index 共同确定，详见功能说明。

## 约束说明

1. index必须是整数类型（DT\_INT32 或 DT\_INT64），值为合法索引，即不能超出 input.shape[dim]；

2. dim为int类型，取值范围：-input.dim <= dim < input.dim。支持负数，负值会被解释为 dim + input.dim；

3. input.shape 的 dim 轴 viewshape 不可切，要求 viewshape\[dim\]\>=input.shape\[dim\]，其余维度的Shape大小不做限制；

4. TileShape的维度与result相同，用于切分result。TileShape 设置需保证 result 不超过UB大小，具体用法详见 [TileShape设置示例]()

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape 的维度设置须与输出张量保持一致，用于控制输出 Tile 块的大小。

以输入$ input[B,S,D]$ 、索引 $index[T]$ 、轴 $	ext{axis}=-2$ 、输出 $output[B,T,D]$  为例：设 TileShape 为$[b_1, t_1, d_1]$，该配置直接作用于输出 output 的各维度，同时映射至输入与索引。其中 $b_1$ 切分 input 的批次维 B ，$d_1$ 切分 input 的特征维 D ，而输入的序列维 S （即轴 −2 ）不参与切分，仅作为索引源； $t_1$ 则作用于索引 index  的长度维 T 。Tile 内存占用须满足约束 $b_1 dot t_1 dot d_1 dot 	ext{sizeof}(athbf{output}) < 	ext{UBSize}$

### 接口调用示例

```python
x = pypto.tensor([3, 4], pypto.DT_FP32)
indices = pypto.tensor([2,], pypto.DT_INT32)
out1 = pypto.index_select(x, 0, indices)
out2 = pypto.index_select(x, 1, indices)
```

结果示例如下：

```python
输入 x:        [[ 0.1427,  0.0231, -0.5414, -1.0009],
                [-0.4664,  0.2647, -0.1228, -1.1068],
                [-1.1734, -0.6571,  0.7230, -0.6004]]
输入 index:    [0, 2]
输出 out1 :    [[ 0.1427,  0.0231, -0.5414, -1.0009],
                [-1.1734, -0.6571,  0.7230, -0.6004]]
输出 out2 :    [[ 0.1427, -0.5414],
                [-0.4664, -0.1228],
                [-1.1734,  0.7230]]
```
