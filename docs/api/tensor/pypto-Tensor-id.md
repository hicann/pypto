# pypto.Tensor.id

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取Tensor的唯一标识

## 函数原型

```python
id(self) -> int
```

## 参数说明

无

## 返回值说明

返回Tensor的唯一标识。

## 约束说明

这是一个只读属性。

## 调用示例

```python
t = pypto.tensor((4, 4), pypto.DT_FP32) 
print(t.id)  # 输出Tensor的ID
```

