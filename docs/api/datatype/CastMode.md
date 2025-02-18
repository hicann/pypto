# CastMode

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

CastMode定义了数据类型转换时的舍入模式，用于控制浮点数转换时的精度处理方式，确保转换结果的准确性。

## 原型定义

```python
class CastMode(enum.Enum):
     CAST_NONE = ...   # 无转换模式，直接截断，不进行舍入处理
     CAST_RINT = ...   # 舍入到最近整数，平局时舍入到偶数
     CAST_ROUND = ...  # 舍入到最近整数，平局时远离零舍入
     CAST_FLOOR = ...  # 向下舍入，向负无穷方向舍入
     CAST_CEIL = ...   # 向上舍入，向正无穷方向舍入
     CAST_TRUNC = ...  # 截断舍入，向零方向舍入
     CAST_ODD = ...    # 舍入到奇数，Von Neumann舍入方式
```

