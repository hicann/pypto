# ReduceMode

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

ReduceMode定义了归约操作的执行模式，用于指定多线程或多设备环境下的归约计算方式，确保计算结果的正确性和性能。

## 原型定义

```python
class ReduceMode(enum.Enum):
     ATOMIC_ADD = ...  # 原子加法归约，使用原子操作确保多线程安全的数据累加
```

