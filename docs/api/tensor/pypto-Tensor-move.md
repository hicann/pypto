# pypto.Tensor.move

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

将一个Tensor数据移动到当前Tensor。

## 函数原型

```python
move(self, other: 'Tensor') -> None
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | 输入      | 要移动数据的源Tensor。 |

## 返回值说明

无

## 约束说明

无。

## 调用示例

```python
t1 = pypto.tensor((2, 3), pypto.DT_FP32)
t2 = pypto.tensor((2, 3), pypto.DT_FP32)
# 将 t2 的数据移动到 t1
t1.move(t2)
```

