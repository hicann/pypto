# RemAlgorithm

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

RemAlgorithm定义了remainder操作的精度模式，用于控制指数计算时的精度处理方式。

## 原型定义

```python
class RemAlgorithm(enum.Enum):
     INTRINSIC = ...       # 指令模式，直接使用芯片指令
     HIGH_PRECISION = ...  # 高精度模式
```

## 参数说明

| 参数值 | 说明 |
|:-------|:-----|
| HIGH_PRECISION | 高精度模式。在底层实现中使用更高精度的计算方式，减少精度损失。 |
| INTRINSIC | 指令模式。直接使用芯片指令进行计算。 |

## 使用建议

1. **默认行为**：如果不指定精度模式，默认使用 `HIGH_PRECISION` 模式，以确保计算精度。
2. **精度要求高的场景**：推荐使用 `HIGH_PRECISION` 模式，可以有效减少精度损失，提高计算结果的准确性。
3. **对精度要求不高的场景**：可以使用 `INTRINSIC` 模式，直接使用芯片指令进行计算。
4. **注意**：`HIGH_PRECISION` 模式仅在Ascend 950PR/Ascend 950DT上有效，其他产品暂时默认使用指令模式 `INTRINSIC`。

## 使用示例

```python
import pypto

# 创建张量
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)

# 使用高精度模式
out = pypto.remainder(a, b, pypto.RemAlgorithm.HIGH_PRECISION)

# 使用指令模式
out = pypto.remainder(a, b, pypto.RemAlgorithm.INTRINSIC)

# 默认使用高精度模式
out = pypto.remainder(a, b)
```
