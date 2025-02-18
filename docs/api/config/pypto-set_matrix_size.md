# pypto.set\_matrix\_size

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

将NZ的输入Tensor在经过reshape后，传入matmul使用计算时，需要将原始的Tensor（未reshape前）的m，k，n值传入，使matmul获取到原始m，k，n值。

## 函数原型

```python
set_matrix_size(size: List[int])-> None
```

## 参数说明


| 参数名  | 输入/输出 | 说明                  |
|---------|-----------|-----------------------|
| size    | 输入      | 输入Tensor的m，k，n值 |

## 返回值说明

void

## 约束说明

1、NZ输入的Tensor，在经过reshape后，调用matmul计算时，需设置该参数。

2、调用matmul的输入是3维/4维的NZ格式Tensor，需设置该参数。

## 调用示例

```python
a = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
pypto.set_matrix_size([32, 64, 16]) #对应输入的Tensor的m，k，n值
out = pypto.matmul(a, b, pypto.DT_FP32)
```

