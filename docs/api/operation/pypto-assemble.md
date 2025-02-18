# pypto.assemble

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

以offsets指定的out索引位置为基准，将输入Tensor input赋值到输出Tensor out的对应区域。

## 函数原型

```python
assemble(input: Tensor, offsets: List[Union[int, SymbolicScalar]], out: Tensor) -> None
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| offsets | 输入      | 相对于目标输出的偏移。 <br> 需要保证offsets小于out的Shape。          |
| out     | 输入      | 目的操作数。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

无返回值，会直接对out进行修改。

## 约束说明

无

## 调用示例

```python
x = pypto.tensor([2, 2], pypto.data_type.DT_FP32) 
out = pypto.tensor([4, 4], pypto.data_type.DT_FP32)
offsets = [0, 0]
pypto.assemble(x, offsets, out) 
```

结果示例如下：

```python
输出数据x: [[1, 1]
            [1, 1]]
输入数据out: [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
输出数据out: [[1, 1, 0, 0],
              [1, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]])
```

