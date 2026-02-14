# pypto.tril

## 产品支持情况

| 产品                                        | 是否支持 |
| :------------------------------------------ | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |

## 功能说明

返回二维张量或者一批张量的下三角部分。结果张量的其他元素被设置为0。

## 函数原型

```python
tril(input: Tensor, diagonal: SymInt = 0) -> Tensor:
```

## 参数说明

| 参数名   | 输入/输出 | 说明                                                                                                                                                                                                                        |
| -------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input    | 输入      | 源操作数。`<br>` 支持的类型为：Tensor。 `<br>` Tensor支持的数据类型为：DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_INT8。 `<br>` 不支持空Tensor；Shape仅支持2-5维；Shape Size不大于2147483647（即INT32_MAX）。 |
| diagonal | 输入      | 源操作数，指定需要考虑的对角线，默认为0。`<br>` SymInt类型。                                                                                                                                                    |

## 返回值说明

输出Shape、数据类型与输入input一致的Tensor。

## 约束说明

详见参数说明。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入intput shape为[m, n]，输出为[m, n], TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([3, 3], pypto.data_type.DT_INT32)        # shape (3, 3)
diagonal = 0
out = pypto.tril(x, diagonal)
```

结果示例如下：

```python
输入数据  x :[[1 2 3],
             [4 5 6],
             [7 8 9]]
输出数据 out:[[1 0 0],
             [4 5 0],
             [7 8 9]]                             # shape (3, 3)
```
