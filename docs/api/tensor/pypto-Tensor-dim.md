# pypto.Tensor.dim

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取Tensor维度。

## 函数原型

```python
dim(self) -> int
```

## 参数说明

无

## 返回值说明

Tensor维度。

## 约束说明

这是一个只读属性。

## 调用示例

```python
t = pypto.tensor((2, 3, 4), pypto.DT_FP32)
out = t.dim
```

结果示例如下：

```python
输出数据out: 3
```

