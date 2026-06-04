# pypto.sub

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

逐元素从input中减去other值。计算公式如下：

$$
res_i = input_i - other_i
$$

## 函数原型

```python
sub(input: Tensor, other: Union[Tensor, float]) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。不同型号支持的数据类型有所差异，详细请参见[约束说明](#约束说明)。<br> 不支持空Tensor；Shape仅支持1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为float以及Tensor类型。 不同型号支持的Tensor数据类型有所差异，详细请参见[约束说明](#约束说明)。<br> 不支持空Tensor；Shape仅支持1-4维；支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input、other相同，Shape为input和other广播后大小。

## 约束说明

1. input 和 other 类型应该相同。
2. other 为数字的时候，不支持隐式转化。
3. other 不支持nan、inf等特殊值。
4. Tensor数据类型说明：
   - Ascend 950PR/Ascend 950DT：DT_INT32，DT_UINT32，DT_FP32，DT_INT16，DT_UINT16，DT_FP16，DT_BF16，DT_UINT8，DT_INT8。
   - Atlas A3 训练系列产品/Atlas A3 推理系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。
   - Atlas A2 训练系列产品/Atlas A2 推理系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：非广播场景，输入input shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

示例2：广播场景，输入input shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1], 则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([2, 3], pypto.DT_FP32)
z = pypto.sub(x, y)
```

结果示例如下：

```python
输入数据x: : [[9.0 9.0 9.0],
              [9.0 9.0 9.0]]
输入数据y:   [[1.0 2.0 3.0],
              [1.0 2.0 3.0]]
输出数据z:   [[8.0 7.0 6.0],
              [8.0 7.0 6.0]]
```
