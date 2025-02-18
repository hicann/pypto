# pypto.get\_runtime\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取runtime配置。

## 函数原型

```python
get_runtime_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## 参数说明

无。

## 返回值说明

返回dict，包含runtime的所有配置项信息。

## 约束说明

无。

## 调用示例

```python
pypto.get_runtime_options()
```

