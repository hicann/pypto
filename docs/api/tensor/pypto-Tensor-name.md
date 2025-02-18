# pypto.Tensor.name

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取或设置Tensor的名称。

## 函数原型

```python
name(self) -> str
name(self, value: str) -> None
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| value   | 输入      | 要设置的Tensor的名称。 |

## 返回值说明

Tensor的名称。

## 约束说明

无。

## 调用示例

```python
t = pypto.tensor((2, 3), pypto.DT_FP32)
n1 = t.name
t.name = "my_tensor"
n2 = t.name
```

结果示例如下：

```python
输出n1: ""
输出n2: "my_tensor"
```

