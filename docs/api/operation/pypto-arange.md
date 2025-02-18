# pypto.arange

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

创建长度为$\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil$的一维Tensor，包含区间 \[start, end\) 内、以 step 为步长的等差数列。

## 函数原型

```
arange(start: Union[int, float] = 0, end: Union[int, float], step: Union[int, float] = 1) -> Tensor
```

## 参数说明


| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| start  | 输入      | 源操作数。 <br> 支持的数据类型为：int32, float32。 <br> 默认值为 0。 |
| end    | 输入      | 源操作数。 <br> 支持的数据类型为：int32, float32。 <br> 该参数不能省略。 |
| step   | 输入      | 源操作数。 <br> 支持的数据类型为：int32, float32。 <br> 默认值为 1。 |

## 返回值说明

返回一维输出Tensor，若输入值存在float数据类型，则输出Tensor数据类型为float，否则为int。

## 约束说明

1. step不能为0，作为浮点数，abs\(step\)\>1e-8;

2. \(end-start\)/step需大于0;

3. 如果 start, end, step 均为 int 输入，则三者均不能超出 int32 范围

4. TileShape和输出output维度一致，均为一维，用于切分output

## 调用示例

```python
y1 = pypto.arange(1.0, 4.0, 0.5)
y2 = pypto.arange(1.0, 4.0)
y3 = pypto.arange(4)
```

结果示例如下：

```python
输出数据y1: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
输出数据y2: [1.0, 2.0, 3.0]
输出数据y3: [0, 1, 2, 3]
```

