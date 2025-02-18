# ScatterMode

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

ScatterMode 定义了 scatter 函数 reduce 模式

## 原型定义

```python
class ScatterMode(enum.Enum):
     None = ...     # 仅做数据搬运
     ADD = ...      # 加法模式
     MULTIPLY = ... # 乘法模式
```

