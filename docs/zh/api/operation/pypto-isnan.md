# pypto.isnan

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

逐元素判断张量中的元素值是否为NaN（非数值）。

当元素为NaN或-NaN时，结果中对应元素位置的值为True，其余为False。

普通有限值、`+0.0`/`-0.0`以及`+inf`/`-inf`均判定为非NaN，对应位置的值为False。

## 函数原型

```python
isnan(self: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| self   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32。 <br> 不支持空Tensor；形状大小支持1-4维；形状大小中对应元素的个数不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出张量，张量的数据类型为布尔类型DT_BOOL，形状大小与输入张量的形状大小一致。

## 约束说明

1. 仅支持DT_FP16，DT_BF16，DT_FP32等数据类型。
2. TileShape以及ViewShape的尾轴必须按照输出张量的类型32B对齐，由于输出张量为布尔类型，因此TileShape以及ViewShape的尾轴必须是32的倍数。
3. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。



## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入input shape为[m, n]，输出为[m, n], TileShape设置为[m1, n1],则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 32)
```

### 接口调用示例

```python
self = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
out = pypto.isnan(self)
```

结果示例如下：

```python
输入数据self: [[1 nan 3],
               [inf 1 1],
               [1 1 -inf]]
输出数据out: [[False True False],
             [False False False],
             [False False False]]
```
