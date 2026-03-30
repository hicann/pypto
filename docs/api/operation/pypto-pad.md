# pypto.pad

## 产品支持情况

| 产品                                        | 是否支持 |
| :------------------------------------------ | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

对输入 Tensor 进行填充（Padding）。

填充大小根据 `pad` 参数从输入 Tensor 的最后一个维度开始，由后向前依次描述。`pad` 参数的格式为 $(pad\_left, pad\_right, pad\_top, pad\_bottom, ...)$。当前实现仅支持对最后两个维度进行常量（Constant）模式的右侧（Right）和底部（Bottom）填充。

## 函数原型

```python
pad(input: Tensor, pad: Sequence[int], mode: str = "constant", value: float = 0.0) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                                                                                                                                           |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | 输入      | 需要进行填充的源操作数。<br> 支持的类型为：Tensor。<br> Tensor支持的数据类型为：DT_FP32、DT_FP16、DT_BF16。<br> 不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。                                                    |
| pad    | 输入      | 填充大小序列。<br> 支持的类型为：tuple 或 list (包含int)。<br> 序列长度 $m$ 必须为偶数，且满足 $\frac{m}{2} \leq$ `input` 的维度数。<br> 格式为：`(pad_left, pad_right, pad_top, pad_bottom, ...)`。                           |
| mode   | 输入      | 填充模式。<br> 支持的类型为：str。<br> 可选值为 `'constant'`、`'reflect'`、`'replicate'` 或 `'circular'`。<br> 默认值：`'constant'`。<br> **注意**：当前仅支持 `'constant'` 模式。                                             |
| value  | 输入      | 当填充模式为常量填充 (`'constant'`) 时的填充值。<br> 支持的类型为：float。<br> 支持任意浮点数值，包括 `-inf`、`inf`、`0.0` 以及其他任意浮点数（如 `1.0`、`-1.0`、`0.5` 等）。 默认值：`0.0`。                                                                                                                                 |

## 返回值说明

返回输出 Tensor，Tensor 的数据类型和 `input` 相同，Shape 为根据 `pad` 参数在对应维度上扩展后的大小。

## 约束说明

1. `pad` 参数的长度必须为2或者4。
2. 当前**仅支持多维情况下在右侧（Right）和底部（Bottom）进行填充，或者1维情况下在右侧（Right）填充**。即 `pad` 序列中向左和向上的填充量必须为 0（例如格式必须为 `(0, pad_right, 0, pad_bottom)` 或者`(0, pad_right)`）。
3. mode当前**仅支持 `'constant'`（常量填充）模式**，其他模式暂不支持。
4. value 支持任意浮点数值，填充值的数据类型会自动转换为与输入 Tensor 一致。
5. 如果 `input` 不是 Tensor 类型，或 `pad` 不是整数序列，将抛出 `TypeError`。

## 调用示例

### TileShape设置示例

说明：调用该 operation 接口前，应通过 `set_vec_tile_shapes` 设置 TileShape。

TileShape 维度应和**输出（填充后的 Shape）**一致。

示例1：输入 `input` shape 为 `[m, n]`，如果对其在 n 轴右侧填充了 `p`，则输出 shape 为 `[m, n+p]`，TileShape 设置为 `[m1, n1]`，则 `m1`, `n1` 分别用于切分输出的 `m`, `n+p` 轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
# 示例：对一个 shape 为 [1, 1, 2, 2] 的 Tensor 进行填充
# 最后一个维度 (右侧) 填充 1
# 倒数第二个维度 (底部) 填充 1
t4d = pypto.tensor([0.0, 1.0, 2.0, 3.0], pypto.DT_FP32)
# 假设内部已将一维数据 reshape 为 [1, 1, 2, 2]

p1 = (0, 1, 0, 1)  # (pad_left=0, pad_right=1, pad_top=0, pad_bottom=1)
out = pypto.pad(t4d, p1, mode="constant", value=0.0)
```

结果示例如下：

```python
# 输入数据 t4d (逻辑 shape 为 [1, 1, 2, 2]): 
[[[[0.0, 1.0],
   [2.0, 3.0]]]]

# 输出数据 out (逻辑 shape 扩展为 [1, 1, 3, 3]): 
[[[[0.0, 1.0, 0.0],
   [2.0, 3.0, 0.0],
   [0.0, 0.0, 0.0]]]]
```