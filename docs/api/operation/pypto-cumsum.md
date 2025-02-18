# pypto.cumsum

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

计算输入Tensor input沿指定维度的累积和。

## 函数原型

```python
cumsum(input: Tensor, dim: int) -> Tensor:
```

## 参数说明


| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_INT16, DT_INT32。 <br> 不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim    | 输入      | 源操作数，指定累加维度。 <br> int 类型。                              |

## 返回值说明

输出Shape、数据类型与输入input一致的Tensor。

## 约束说明

1. dim：指定计算累积和的维度，必须在输入Tensor input的有效维度范围内，其值需满足-input.dim <= dim < input.dim，对应轴不切分。

## 调用示例

```python
input = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
dim = 0
y = pypto.cumsum(input, dim)
```

结果示例如下：

```python
输入数据 x:   [[0 1 2],
               [3 4 5]]
输出数据 y:   [[0 1 2],
               [3 5 7]]                             # shape (2, 3)
```

