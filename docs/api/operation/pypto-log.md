# pypto.log

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

对input做以e为底的对数运算

## 函数原型

```python
log(input: Tensor) -> Tensor:
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16, DT_BF16。 <br> 支持的维度：1-4维 <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape为input大小。

## TileShape设置示例

TileShape维度应和输出一致。

如输入intput shape为[m, n]，输出为[m, n], TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## 调用示例

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log(x)
```

结果示例如下：

```python
输入数据x: [1.0     2.0    3.0]
输出数据y: [[0.0000 0.6931 1.0986]
```

