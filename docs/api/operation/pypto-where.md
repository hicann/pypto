# pypto.where

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

condition 为一个布尔类型的掩码张量（mask tensor）。对于张量中任意位置的元素，该操作基于布尔掩码张量 condition 进行逐元素选择。其计算行为可形式化表示为如下表达式。

$$
result_{i}=
\begin{cases}
input_{i} & \text{if } condition_{i}==True \\
other_{i} & \text{if } condition_{i}==False
\end{cases}
$$

condition 须为Tensor，input 和 other 可以为 Tensor、 float  以及 Element，广播规则如下（只支持单轴广播）：

1.  input, other, condition 均为Tensor，result 的 Shape 由三者广播得到。

    例：input:\[1,20,20\], other:\[20,1,20\], condition:\[20,20,1\], result:\[20,20,20\]

2.  只有 input, condition 为 Tensor时，result 的 Shape 由两者广播得到。

    例：input:\[1,20,20\], condition:\[20,20,1\], result:\[20,20,20\]

3.  只有 other, condition 为 Tensor时，result 的 Shape 由两者广播得到。

    例：other:\[20,1,20\], condition:\[20,20,1\], result:\[20,20,20\]

4.  只有 condition 为 Tensor时，result 的 Shape 与 condition 一致。

## 函数原型

```python
where(
    condition: Tensor,
    input: Union[Tensor, float, Element],
    other: Union[Tensor, float, Element]
) -> Tensor
```

## 参数说明


| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| condition   | 输入      | 支持的类型为：Tensor。<br> Tensor支持的数据类型为：DT_BOOL。<br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。<br> 作为条件选择input或者other的元素。 |
| input       | 输入      | 支持的类型为 float\Element\Tensor类型。<br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。<br> Tensor和Element支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。<br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other       | 输入      | 支持的类型为 float\Element\Tensor类型。<br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。<br> Tensor和Element支持的数据类型为：DT_FP32，DT_FP16，DT_BF16。<br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

result ：Tensor，Shape由输入的广播得到，详细广播场景可看上文。数据类型和input、other保持一致。

## 约束说明

1. 建议优先使用 Element，传入 float 标量对于 fp16 场景，不保证正确性。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：非广播场景，输入condition为[m, n]，input为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

示例2：广播场景，输入condition为[m, 1]，input为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
cond1 = pypto.tensor([4], pypto.DT_BOOL)
a1 = pypto.tensor([4], pypto.DT_FP32)
b1 = pypto.tensor([4], pypto.DT_FP32)
out1 = pypto.where(cond1, a1, b1)

# Using scalar inputs
out2 = pypto.where(cond1, 1, 0) 

# Broadcasting example
cond2 = pypto.tensor([2, 2], pypto.DT_BOOL) 
a2 = pypto.tensor([2], pypto.DT_FP32)  
b2 = 0.0
out3 = pypto.where(cond2, a2, b2)
```

结果示例如下：

```python
输入数据cond1: [True, False, True, False]
输入数据a1:    [1.0  2.0  3.0  4.0] 
输入数据b1:    [10.0 20.0 30.0 40.0] 
输出数据out1:  [1.0  20.0 3.0  40.0]

输出数据out2:  [1.0 0.0 1.0 0.0]

输入数据cond2 = [[True, False], [False, True]]
输入数据a2:      [1.0 2.0]
输出数据out3:   [[1.0 0.0], 
                 [0.0 2.0]]
```

