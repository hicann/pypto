# pypto.sign

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

计算输入Tensor中每个元素的符号值，逐元素运算。计算公式如下：

$$
\text{sign}(x) = \begin{cases}
-1 & \text{if } x < 0 \\
0 & \text{if } x = 0 \\
1 & \text{if } x > 0
\end{cases}
$$

## 函数原型

```python
sign(input: Tensor) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32，DT_INT8，DT_INT16，DT_INT32。 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回Tensor类型。其Shape与输入Tensor一致，数据类型与输入Tensor一致，其元素为输入Tensor对应元素的符号值（-1、0或1）。

## 约束说明

1.  TileShape与input维度保持一致；
2.  由于存在临时内存使用，当输入数据类型为DT\_INT8，TileShape大小有额外约束，假设TileShape为\[a,b,c,d\]，那么a\*b\*c\*d\*sizeof\(self\) + c\*d\*sizeof\(INT8\) <UB。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入input shape为[m, n]，输出为[m, n], TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.sign(x)
```

结果示例如下：

```python
输入数据x: [-5.0, 0.0, 5.0, 10.0, -2.0]
输出数据y: [-1.0, 0.0, 1.0, 1.0, -1.0]
```
