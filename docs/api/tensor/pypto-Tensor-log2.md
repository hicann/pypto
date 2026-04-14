# pypto.Tensor.log2

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 函数原型

```python
log2(self, precision_type: LogAlgorithm = LogAlgorithm.INTRINSIC) -> 'Tensor'
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| precision_type  | 输入      | 精度类型。 <br> 支持的类型为：LogAlgorithm。 <br> 默认值为 LogAlgorithm.INTRINSIC。 <br> INTRINSIC 直接使用芯片指令进行计算，速度更快；HIGH_PRECISION 使用更高精度的计算以减少精度损失。 |

## 详细说明

请参见[pypto.log2](../operation/pypto-log2.md)。