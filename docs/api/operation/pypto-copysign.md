# pypto.add

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

逐元素地取input的数值和other的符号。计算公式如下：

$$
res_i = input_i * sign(other_i)
$$

## 函数原型

```python
copysign(input: Tensor, other: Tensor) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为Tensor类型。 <br> Tensor支持的数据类型为：DT_FP32/DT_FP16/DT_BF16。 <br> 不支持空Tensor；Shape仅支持2-4维，支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为Tensor类型。 <br> Tensor支持的数据类型为：DT_FP32/DT_FP16/DT_BF16。 <br> 不支持空Tensor；Shape仅支持2-4维，支持按照单个维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input、other相同，Shape为input和other广播后大小。

## 约束说明

1.  input 和 other 类型应该相同。

## 调用示例

```python
a = pypto.tensor([3, 3], pypto.DT_FP32)
b = pypto.tensor([3, 3], pypto.DT_FP32)
out = pypto.copysign(a, b)
```

结果示例如下：

```python
Input  x : [[1 -2  3],
           [4  5 -6],
           [-7 8  9]]
Input  y : [[-1 6 -8],
            [1 -1  0],
            [7 -8  9]]
Output out:[[-1 2 -3],
            [4 -5  6],
            [7 -8  9]]
```