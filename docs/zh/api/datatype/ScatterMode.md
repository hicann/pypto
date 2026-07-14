# ScatterMode

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

ScatterMode定义了scatter函数reduce模式

## 原型定义

```python
class ScatterMode(enum.Enum):
     None = ...     # 仅做数据搬运
     ADD = ...      # 加法模式
     MULTIPLY = ... # 乘法模式
```
