# pypto.topk

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取最后一个维度的前k个最大值或最小值及其对应的索引。

如果输入是向量，则在向量中找到前k个最大值或最小值及其对应的索引；如果输入是矩阵，则沿最后一个维度计算每行中前k个最大值或最小值及其对应的索引。如下图所示，对Shape为\(4, 32\)的二维矩阵进行排序，k设置为1，输出结果为\[\[32\] \[32\] \[32\] \[32\]\]。

![](../figures/nz-reduce.png)

## 函数原型

```python
topk(input: Tensor, k: int, dim: Optional[int]=None, largest: bool=True) -> Tuple[Tensor, Tensor]
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br> 支持的类型为：Tensor。<br> Tensor支持的数据类型为：DT_FP32。<br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| k       | 输入      | 返回元素的数量。<br> k的大小应该满足：1 <= k <= input.shape[dim]。 |
| dim     | 输入      | 指定排序的维度。<br> 目前仅支持按最后一个维度排序，即dim= -1或dim= input.shape.size() - 1。 |
| largest | 输入      | 如果为True，返回最大元素。如果为False，返回最小元素。 |

## 返回值说明

返回一个命名元组(values, indices)，其中包含input在指定维度dim下每行中最大或最小的k个元素的值和索引。

## 约束说明

1.  只支持对尾轴进行topk操作；
2.  TileShape尾轴32bytes对齐\(TileShape\[-1\]\*4 % 32 == 0\)，且TileShape尾轴需要小于22KB\(TileShape\[-1\]\*4 < 22KB\)；
3.  k <= TileShape\[-1\] && k <= input.shape\[-1\]；

## 调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.topk(x, 2, -1, True)
```

结果示例如下：

```python
输入数据x: [[1.0 2.0 3.0], 
            [1.0 2.0 3.0]]
输出数据y[0]: [[3.0 2.0], 
               [3.0 2.0]]
输出数据y[1]: [[2, 1],
               [2, 1]]
```

