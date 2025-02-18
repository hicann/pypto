# ReLuType

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

ReLuType定义了RELU激活函数的模式，用于启用RELU功能。

## 原型定义

```python
class ReLuType(enum.Enum):
     NO_RELU= ...  # 不使能ReLu功能
     RELU= ...     # 使能ReLu功能
```

