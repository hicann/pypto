# pypto.sum

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

对一个多维向量按照指定的维度进行数据累加。

定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对Shape为\(2, 3\)的二维矩阵进行运算，指定在第一维计算数据的累加，输出结果为\[5, 7, 9\]；指定在第二维计算数据的累加，输出结果为\[6, 15\]。

**图 1**  sum按第一个维度计算示例  
![](../figures/sum按第一个维度计算示例.png "sum按第一个维度计算示例")

**图 2**  sum按最后一个维度计算示例  
![](../figures/sum按最后一个维度计算示例.png "sum按最后一个维度计算示例")

## 函数原型

```python
sum(input: Tensor,  dim: int, keepdim: bool = False) -> Tensor: 
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维，Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 源操作数。 <br> 支持任意单轴。 |
| keepdim | 输入      | 源操作数。 <br> 控制在进行归约后，是否保持被压缩的维度。 <br> 默认值为False。 |

## 返回值说明

返回输出Tensor，输出Tensor的Shape与keepdim参数相关。

若keepdim参数为 True，则在执行归约操作后保留被归约的维度。输出Tensor在除dim指定的维度外，其他维度的Shape与输入Tensor的Shape一致，而在dim指定的维度上的大小为 1。

若keepdim参数为 False（默认），则被归约的维度会从输出Tensor中移除。

## 约束说明

1. TileShape大小不超过 64KB；

2. 尾轴要 32bytes 对齐;

3. TileShape次尾轴要小于等于255，即 TileShape\[-2\]<=255.

## 调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.sum(x, -1, True)
```

结果示例如下：

```
输入数据 x: [[1.0 2.0 3.0], 
             [1.0 2.0 3.0]]
输出数据 y: [[6.0],
             [6.0]]
```

