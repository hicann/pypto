# pypto.fillpad

## 产品支持情况

| 产品                                        | 是否支持 |
| :------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

对输入 Tensor 进行填充（Padding）。

和pad不同，此接口不会改变张量的形状，他将填充区域(即超过validshape的区域)用指令的值进行填充。当前实现支持输入1-2维tensor， 进行常量（Constant）模式的右侧（Right）和底部（Bottom）填充。

## 函数原型

```python
fillpad(input: Tensor, mode: str = "constant", value: float = 0.0) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                                                                                                                                           |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | 输入      | 需要进行填充的源操作数。<br> 支持的类型为：Tensor。<br> Tensor支持的数据类型为：DT_FP32、DT_FP16、DT_BF16。<br> 不支持空Tensor；Shape支持1-2维；Shape Size不大于2147483647（即INT32_MAX）。                                                    |
| mode   | 输入      | 填充模式。<br> 支持的类型为：str。<br> 可选值为 `'constant'`、`'reflect'`、`'replicate'` 或 `'circular'`。<br> 默认值：`'constant'`。<br> **注意**：当前仅支持 `'constant'` 模式。                                             |
| value  | 输入      | 当填充模式为常量填充 (`'constant'`) 时的填充值。<br> 支持的类型为：float。<br> 支持任意浮点数值，包括 `-inf`、`inf`、`0.0` 以及其他任意浮点数（如 `1.0`、`-1.0`、`0.5` 等）。 默认值：`0.0`。                                                                                                                                 |

## 返回值说明

返回输出 Tensor，Tensor 的数据类型和`input` 相同，Shape 也和`input` 相同。

## 约束说明

1. mode当前**仅支持 `'constant'`（常量填充）模式**，其他模式暂不支持。
2. value 支持任意浮点数值，填充值的数据类型会自动转换为与输入 Tensor 一致。
3. 如果 `input` 不是 Tensor 类型，将抛出 `TypeError`。

## 调用示例

### TileShape设置示例

说明：调用该 operation 接口前，应通过 `set_vec_tile_shapes` 设置 TileShape。

TileShape 维度应和**输出**一致。

示例1：输入 `input` shape 为 `[m, n]`，则输出 shape 为 `[m, n]`，TileShape 设置为 `[m1, n1]`，则 `m1`, `n1` 分别用于切分输出的 `m`, `n` 轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
a = pypto.tensor([4, 4], pypto.DT_FP32)
out = pypto.fillpad(a, "constant", "-inf")
```

结果示例如下：

```python
# 输入数据 t4d (逻辑 shape 为 [4, 4]):
[[1.0, 2.0, 0.0, 0.0],
[3.0, 4.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0]]

# 输出数据 out (逻辑 shape 为 [4, 4]):
[[1.0, 2.0, -inf, -inf],
[3.0, 4.0, -inf, -inf],
[-inf, -inf, -inf, -inf],
[-inf, -inf, -inf, -inf]]
```
