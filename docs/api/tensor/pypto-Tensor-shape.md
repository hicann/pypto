# pypto.Tensor.shape

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取Tensor Shape。

## 函数原型

```python
shape(self) -> List[SymInt]
```

## 参数说明

无

## 返回值说明

返回Tensor的形状列表。

## 约束说明

无。

## 调用示例

```python
t = pypto.tensor((16, 32), pypto.DT_FP32)
out = t.shape
```

结果示例如下：

```python
输出数据out: [16, 32]
```

