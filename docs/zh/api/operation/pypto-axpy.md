# pypto.axpy_

## 产品支持情况

| 产品                                        | 是否支持 |
| :------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

执行 AXPY 操作：`y = alpha * x + y`。该操作对 y 张量进行原地更新。

计算公式如下：

$$
y_i = \alpha \cdot x_i + y_i
$$

**重要说明**：AXPY 是原地操作，y 张量会被直接修改。如果后续计算需要使用执行 AXPY 之前的原始 y 值，请在调用 AXPY 前使用 `pypto.clone(y)` 进行备份。

## 函数原型

```python
axpy_(y: Tensor, x: Tensor, alpha: Union[int, float] = 1.0) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                                                                                                                                           |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| y      | 输入/输出 | 目标张量，将被原地更新。<br> 支持的类型为：Tensor。<br> Tensor支持的数据类型为：DT_FP32、DT_FP16。<br> **不支持广播**：y 的形状必须能容纳 x 的广播结果，即 y 的任意维度不能为 1（除非 x 对应维度也为 1）。<br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| x      | 输入      | 源张量，可广播到 y 的形状。<br> 支持的类型为：Tensor。<br> Tensor支持的数据类型为：DT_FP32、DT_FP16。<br> 支持广播：x 可以广播到 y 的形状（如 x 形状为 `[m, 1]`，y 形状为 `[m, n]`）。<br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。                                                 |
| alpha  | 输入      | 缩放因子，用于对 x 进行缩放。<br> 支持的类型为：int、float，默认值为 1.0。<br> alpha 的数据类型会自动转换为与 y 一致。                                                                                                            |

## 返回值说明

返回更新后的 y 张量（与输入 y 共享同一内存地址），Tensor 的数据类型与 y 相同，Shape 与 y 相同。

## 约束说明

1. **dtype 约束**：
   - 相同 dtype：支持 DT_FP32 + DT_FP32、DT_FP16 + DT_FP16。
   - 混合 dtype：仅支持 DT_FP32 (y) + DT_FP16 (x)，其他组合不支持。
2. **广播约束**：
   - y 张量**不支持广播**。如果 y 的某个维度为 1 而 x 对应维度不为 1，将报错。
   - x 张量**支持广播**到 y 的形状。
3. **Shape 约束**：y 和 x 的维度数必须相同（1-4维）。
4. **Format 约束**：y 和 x 的 Format 必须一致。
5. **原地更新注意**：由于 AXPY 是原地操作，y 的原始值会被覆盖。如需保留原始 y 值，请提前 clone：

```python
y_backup = pypto.clone(y)  # 备份原始 y 值
y.axpy_(x, alpha=2.0)      # y 被原地更新
# 此时 y_backup 仍保留原始值，可用于后续计算
```

## 调用示例

### TileShape设置示例

调用该 operation 接口前，应通过 `set_vec_tile_shapes` 设置 TileShape。

TileShape 维度应和输出一致。

示例：输入 y shape 为 `[m, n]`，x shape 为 `[m, n]`（或 `[m, 1]` 广播场景），输出 shape 为 `[m, n]`，TileShape 设置为 `[m1, n1]`，则 `m1`, `n1` 分别用于切分输出的 `m`, `n` 轴。

```python
pypto.set_vec_tile_shapes(32, 32)
```

### 接口调用示例

#### 基本用法

```python
y = pypto.tensor([1, 3], pypto.DT_FP32)
x = pypto.tensor([1, 3], pypto.DT_FP32)
y.axpy_(x, alpha=2.0)
```

结果示例如下：

```python
输入数据 y:   [[1.0 2.0 3.0]]
输入数据 x:   [[2.0 3.0 4.0]]
alpha:        2.0
输出数据 y:   [[5.0 8.0 11.0]]  # y = 2.0 * x + y
```

#### 广播场景

```python
y = pypto.tensor([64, 64], pypto.DT_FP32)  # y shape: [64, 64]
x = pypto.tensor([64, 1], pypto.DT_FP32)   # x shape: [64, 1] (广播到 [64, 64])
y.axpy_(x, alpha=1.5)
```

#### 保留原始 y 值

```python
y = pypto.tensor([32, 32], pypto.DT_FP32)
x = pypto.tensor([32, 32], pypto.DT_FP32)

# 如需使用原始 y 值，提前备份
y_backup = pypto.clone(y)

# 执行 AXPY，y 被原地更新
y.axpy_(x, alpha=2.0)

# y_backup 仍保留原始值，可用于其他计算
diff = pypto.sub(y, y_backup)  # 计算 y 与原始值的差值
```

#### 混合精度（FP32 + FP16）

```python
y = pypto.tensor([32, 32], pypto.DT_FP32)  # y 为 FP32
x = pypto.tensor([32, 32], pypto.DT_FP16)  # x 为 FP16
y.axpy_(x, alpha=1.0)  # 支持 FP32(y) + FP16(x)
```

#### 一维场景

```python
y = pypto.tensor([128], pypto.DT_FP32)
x = pypto.tensor([128], pypto.DT_FP32)
pypto.set_vec_tile_shapes(64)
y.axpy_(x, alpha=2.0)
```
