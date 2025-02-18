# pypto.Element.dtype

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取数据类型。

## 函数原型

```python
def dtype(self) -> pypto.DataType
```

## 参数说明

NA

## 返回值说明

返回Element类型。

## 约束说明

只读数据。

## 调用示例

```python
t = pypto.element(pypto.DT_FP32, 3)
t.dtype
```

