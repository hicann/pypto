# ExpAlgorithm

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

ExpAlgorithm定义了指数操作的精度模式，用于控制指数计算时的精度处理方式。

## 原型定义

```python
class ExpAlgorithm(enum.Enum):
     INTRINSIC = ...       # 指令模式，直接使用芯片指令
     HIGH_PRECISION = ...  # 高精度模式
```

## 参数说明

| 参数值 | 说明 |
|:-------|:-----|
| HIGH_PRECISION | 高精度模式。在底层实现中使用更高精度的计算方式，减少精度损失。 |
| INTRINSIC | 指令模式。直接使用芯片指令进行计算。 |

## 使用建议

1. **默认行为**：如果不指定精度模式，默认使用 `INTRINSIC` 模式，直接使用芯片指令进行计算，速度更快。
2. **精度要求高的场景**：推荐使用 `HIGH_PRECISION` 模式，可以有效减少精度损失，提高计算结果的准确性。
3. **对速度要求高的场景**：使用 `INTRINSIC` 模式，直接使用芯片指令进行计算，速度更快。

## 使用示例

```python
import pypto

# 创建张量
x = pypto.tensor([3], pypto.DT_FP16)

# 使用高精度模式
y = pypto.exp(x, pypto.ExpAlgorithm.HIGH_PRECISION)

# 使用指令模式
y = pypto.exp(x, pypto.ExpAlgorithm.INTRINSIC)

# 默认使用指令模式
y = pypto.exp(x)
```