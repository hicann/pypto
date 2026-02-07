# pypto.one\_hot

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

将整数Tensor转换为对应的 one-hot 编码，其中每个整数被转换为一个向量，只有对应位置为1，其余为0。

## 函数原型

```python
one_hot(input: Tensor, num_classes: int) -> Tensor
```

## 参数说明


| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| input       | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_INT8, DT_INT16, DT_INT32, DT_INT64。 <br> 支持维度1-3维 <br> 内部元素需为非负数。 <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| num_classes | 输入      | one-hot编码长度。 <br> 需大于input中最大元素。 |

## 返回值说明

返回一个Shape为\(input, num\_classes\)、数据类型为DT\_INT64的Tensor。

## 约束说明

TileShape 对输出切分，TileShape 的维度与输出一致，TileShape 的尾轴需等于 num\_classes 。

## TileShape设置示例

TileShape维度应和输出一致。

如输入intput shape为[m, n]，输出为[m, n, t], 其中t=num\_classes，TileShape设置为[m1, n1, t1], 则m1, n1分别用于切分m, n轴。t1必须等于 num\_classes, t轴不可切，必须保证t轴全载。

```python
pypto.set_vec_tile_shapes(m1, n1, t1)
```

## 调用示例

```python
x = pypto.tensor([3], pypto.DT_INT32)
y = pypto.one_hot(x, 5)
```

结果示例如下：

```python
输入数据x: [0, 2, 4]
输出数据y: [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]
```

