# pypto.zeros

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

创建一个大小为 `size`、填充值全为 `0` 的Tensor。其数据类型由 `dtype` 指定，默认数据类型为 `DT_FP32`。

## 注意事项

- **必须先设置 TileShape**：调用此接口前，必须先通过 [set_vec_tile_shapes](pypto-set_vec_tile_shapes.md) 设置 TileShape

## 函数原型

```python
zeros(*size: Union[int, Sequence[int]], dtype: Optional[DataType] = None) -> Tensor
```

## 参数说明

| 参数名       | 输入/输出 | 说明                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| *size        | 输入      | 源操作数，用于定义输出Tensor的Shape。<br> 支持可变长参数（多个int）或单一的序列（如 List[int] 或 Tuple[int]）。 |
| dtype        | 输入      | 源操作数，可选参数，用于定义输出Tensor的数据类型。<br> 支持的数据类型为：`DT_FP32`，`DT_INT32`，`DT_INT16`，`DT_FP16`，`DT_BF16`。<br> 默认值为 `pypto.DT_FP32`。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型由 `dtype` 决定，Shape为 `size` 大小，全部的值均为 `0`。

## 约束说明

1. `tileshape` 的维度需要与输出 result 维度相同，用于切分 result。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过 `set_vec_tile_shapes` 设置TileShape。TileShape维度应和输出一致。
如输入size为 `[m, n]`，输出为 `[m, n]`，TileShape设置为 `[m1, n1]`，则 `m1`, `n1` 分别用于切分 `m`, `n` 轴。

```python
pypto.set_vec_tile_shapes(2, 3)
```

### 接口调用示例

```python
# 示例1：使用可变参数传入size，使用默认dtype (DT_FP32)
x1 = pypto.zeros(2, 3)

# 示例2：使用元组传入size，显式指定dtype (DT_INT32)
x2 = pypto.zeros((2, 3), dtype=pypto.DT_INT32)
```

结果示例如下：

```python
x1输出数据: [[0., 0., 0.], 
             [0., 0., 0.]]
x2输出数据: [[0, 0, 0], 
             [0, 0, 0]]
```
