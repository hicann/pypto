# pypto.LReLU

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

对输入张量逐元素应用 Leaky ReLU（带泄漏的线性整流函数）激活函数。计算公式如下：

$$
\text{res}_i = 
\begin{cases} 
\text{input}_i & \text{if } \text{input}_i \geq 0 \\
\text{negative\_slope} \cdot \text{input}_i & \text{if } \text{input}_i < 0 
\end{cases}
$$

其中 `negative_slope` 为负斜率参数，默认值为 `0.01`。


## 函数原型

```python
LReLU(input: Tensor, negative_slope: float = 0.01) -> Tensor
```

## 参数说明

| 参数名         | 输入/输出 | 说明                                                                 |
|----------------|-----------|----------------------------------------------------------------------|
| input          | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32。<br>不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| negative_slope | 输入      | 负区间的斜率系数。<br>类型为 float，默认值为 `0.01`。<br>必须为非负实数（≥ 0），不支持 `nan`、`inf` 等特殊值。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape与input一致。

## 约束说明

1.  input 数据类型必须为 DT_FP16、DT_BF16 或 DT_FP32。
2.  negative_slope 必须为非负浮点数（≥ 0），且不能为 `nan` 或 `inf`。
3.  不支持 in-place 操作（即输出不能与输入共享内存）。

## TileShape设置示例

TileShape维度应和输出一致（与输入相同）。

例如，输入 shape 为 `[m, n]`，输出也为 `[m, n]`，TileShape 设置为 `[m1, n1]`，则 `m1`、`n1` 分别用于切分 `m`、`n` 轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```


## 调用示例

```python
a = pypto.tensor([[-1.0, 0.0, 1.0]], pypto.DT_FP32)
out = pypto.LReLU(a)
```

结果示例如下：

```python
输入数据a:   [[-1.0  0.0  1.0]]
输出数据out: [[-0.01  0.0   1.0]]
```

