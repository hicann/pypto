# pypto.Tensor.div

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 函数原型

```python
div(self, other: 'Tensor | int | float', div_algorithm: DivAlgorithm = DivAlgorithm.HIGH_PRECISION) -> 'Tensor'
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | 输入      | 除数。 <br> 支持的类型为：Tensor、int、float。 |
| div_algorithm  | 输入      | 精度算法。 <br> 支持的类型为：DivAlgorithm。 <br> 默认值为 DivAlgorithm.HIGH_PRECISION。 <br> HIGH_PRECISION 使用更高精度的计算以减少精度损失；INTRINSIC 直接使用芯片指令。 |

## 详细说明

请参见[pypto.div](../operation/pypto-div.md)。
