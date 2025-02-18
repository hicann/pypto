# pypto.Tensor.format

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取Tensor的格式。

## 函数原型

```python
format(self) -> TileOpFormat
```

## 参数说明

无

## 返回值说明

TileOpFormat：返回Tensor的格式。

## 约束说明

这是一个只读属性。

## 调用示例

```python
t = pypto.tensor((4, 4), pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND) 
print(t.format) 
```

结果示例如下：

```python
输出：TileOpFormat.TILEOP_ND
```

