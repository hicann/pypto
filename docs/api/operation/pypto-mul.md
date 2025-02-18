# pypto.mul

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

按元素求积，公式表达如下：

$$
res_{i} = input_{i} * other_{i}
$$

## 函数原型

```python
mul(input: Tensor, other: Union[Tensor, float]) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维，并支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为float以及Tensor类型。 <br> Tensor支持的数据类型为：DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维，并支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input、other相同，Shape为input和other广播后大小。

## 约束说明

1.  input 和 other 类型应该相同。
2.  other 为数字的时候，不支持隐式转化。
3.  other 不支持nan、inf等特殊值

## 调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([2, 3], pypto.DT_FP32)
z = pypto.mul(x, y)
```

结果示例如下：

```python
输入数据x:  [[1.0 2.0 3.0], 
             [1.0 2.0 3.0]]
输入数据y:  [[1.0 2.0 3.0], 
             [1.0 2.0 3.0]]
输出数据z:  [[1.0 4.0 9.0], 
             [1.0 4.0 9.0]]
```

