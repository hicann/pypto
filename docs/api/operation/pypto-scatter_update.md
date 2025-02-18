# pypto.scatter\_update

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

功能1：将4维src根据2维索引index更新到4维 input 上，计算公式如下：

$$
input\left[\frac{\text{index}[i][j]}{\text{blockSize}}\right]\left[\text{index}[i][j] \% \text{blockSize}\right][0][\dots] = src[i][j][0][\dots]
$$

功能2：将2维src根据2维index更新到2维 input 上，计算公式如下：

$$
input[\text{index}[i][j][\dots]] = src[i*s + j][\dots]
$$

## 函数原型

```python
scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16, DT_BF16, INT32, INT16。 <br> 支持的维度：2维，4维 <br> 2维Shape[b * s, d]，4维Shape[b, s, 1, d] <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 请保持默认值-2。 |
| index   | 输入      | input的一组索引。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：INT64, INT32, INT16。 <br> 支持的维度：2维 <br> Shape[b, s] |
| src     | 输入      | src是一组更新值。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16, DT_BF16, INT32, INT16。数据类型和input保持一致 <br> 支持的维度：2维，4维 <br> 2维Shape[blockNum * blockSize, d]，4维Shape[blockNum, blockSize, 1, d] <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回更新后的 input，为inplace操作。

## 约束说明

broadcast 约束：不支持 broadcast。

ViewShape 约束：2维场景下ViewShape为\[viewB \* s, d\]，4维场景下ViewShape为\[viewB, viewS, 1, d\]，尾轴d不做切分。2维场景下，ViewShape针对src做切分，其第0维是index的第1维s的倍数，\[viewB, S\]针对index做切分。4维场景下，ViewShape针对src做切分，\[viewB, viewS\]针对index做切分。

TileShape 约束：2维场景下TileShape为\[tileS, d\]，4维场景下TileShape为\[tileB, tileS, 1, d\]。尾轴d不做切分。2维场景下，TileShape针对src做切分，\[1，tileS\]针对index做切分，tileBS是输入index的第1维s的约数，如src为\[12, 64\]，index为\[3, 4\]，TileShape为\[TileS, 64\]，其中，TileS可以是1、2、4。4维场景下，TileShape针对src做切分，并且，\[tileB, tileS\]针对index做切分。由于TileShape的切分针对src和index，切块大小之和应小于UB限制。

## 调用示例

-   将2维 src 根据2维index更新到2维input上

    ```python
    x = pypto.tensor([8, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_INT32)
    o = pypto.scatter_update(x, -2, y, z)
    ```

    结果示例如下：

    ```python
    输入数据x:[[0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0]]
    输入数据y:[[1 2],
               [4 5]]
    输入数据z:[[1 2 3],
               [4 5 6],
               [7 8 9],
               [10 11 12]]
    输出数据o:[[0 0 0],
               [1 2 3],
               [4 5 6],
               [0 0 0],
               [7 8 9],
               [10 11 12],
               [0 0 0],
               [0 0 0]])
    ```

-   将4维src根据2维索引index更新到4维input上

    ```python
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_INT32)
    o = pypto.scatter_update(x, -2, y, z)
    ```

    结果示例如下：

    ```python
    输入数据x:[[
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
               ],
               [
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
               ]]
    输入数据y:[[1 8],
               [4 10]]
    输入数据z:[[
                 [[1 2 3]],
                 [[4 5 6]],
               ],
               [
                 [[7 8 9]],
                 [[10 11 12]],
               ]]
    输出数据o:[[
                 [[0 0 0]],
                 [[1 2 3]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[7 8 9]],
                 [[0 0 0]],
               ],
               [
                 [[0 0 0]],
                 [[0 0 0]],
                 [[4 5 6]],
                 [[0 0 0]],
                 [[10 11 12]],
                 [[0 0 0]],
               ]]
    ```

