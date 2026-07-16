# pypto.experimental.get\_operation\_options

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

获取operation配置。

## 函数原型

```python
get_operation_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## 参数说明

无。

## 返回值说明

返回dict，包含operation的所有配置项信息。

## 约束说明

1. 返回值为Dict类型，包含operation的所有配置项信息。
2. 不同配置项的类型可能不同：str、int、List[int]、Dict[int, int] 等。

## 调用示例

```python
pypto.get_operation_options()
```
