# pypto.exp

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

计算输入Tensor中每个元素的 e 的指数，逐元素运算，返回与输入形状相同的Tensor。

## 接口原型

```python
pypto.exp(input, precision_type=pypto.ExpAlgorithm.INTRINSIC) -> Tensor
```

## 参数说明

| 参数 | 类型 | 说明 |
|:-----|:-----|:-----|
| input | Tensor | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32。 <br> 不支持空Tensor；支持的维度：1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | ExpAlgorithm, 可选 | 指数操作的精度模式。默认值为 `ExpAlgorithm.INTRINSIC`。<br>**INTRINSIC**：直接使用芯片指令进行计算，速度更快。<br>**HIGH_PRECISION**：使用更高精度的计算方式，减少精度损失。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape与input相同。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.exp(x)
```

结果示例如下：

```python
输入数据x: [0.0    1.0    2.0]
输出数据y: [1.0000  2.7183  7.3891]
```

### 高精度模式示例

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.exp(x, pypto.ExpAlgorithm.HIGH_PRECISION)
```

### 低精度模式示例

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.exp(x, pypto.ExpAlgorithm.INTRINSIC)
```
