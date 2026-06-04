# pypto.asinh

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

逐元素计算输入 Tensor 的反双曲正弦值。

$$
y_i = \operatorname{asinh}(x_i) = \ln(x_i + \sqrt{x_i^2 + 1})
$$

## 函数原型

```python
asinh(input: Tensor) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor 支持的数据类型为：DT_FP32, DT_FP16, DT_BF16。<br>不支持空 Tensor；Shape 仅支持 1-4 维；Shape Size 不大于 2147483647（即 INT32_MAX）。 |

## 返回值说明

返回输出 Tensor，Shape 与 `input` 相同，数据类型与 `input` 相同，元素值为输入 Tensor 对应元素的反双曲正弦值。

## 约束说明

1. 考虑输入、输出及临时空间占用，TileShape大小有额外约束。假设TileShape为\[a,b,c,d\]，记 $d_{align}=CeilAlign(d, 8)$, $k=d_{align}/8$，$p=\lceil8/k\rceil$，$c_{pad}=c+p-1$，则总的UB空间占用为：

   $$
   a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)+5*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$

## 调用示例

### TileShape 设置示例

TileShape 维度应和输出一致。

如输入 `input` shape 为 `[m, n]`，输出为 `[m, n]`，TileShape 设置为 `[m1, n1]`，则 `m1`、`n1` 分别用于切分 `m`、`n` 轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.asinh(x)
```

结果示例如下：

```python
输入数据 x: [0.0000, 1.0000, 2.0000, -1.0000]
输出数据 y: [0.0000, 0.8814, 1.4436, -0.8814]
```
