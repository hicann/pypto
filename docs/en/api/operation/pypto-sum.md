# pypto.sum

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:57:23.820Z pushedAt=2026-09-05T07:36:26.376Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Accumulates data of a multi-dimensional tensor along a specified dimension.

Define the specified computation dimension as the R-axis (Reduce axis) and the non-specified dimension as the A-axis (Normal axis). As shown in the figure below, for a two-dimensional matrix of shape \(2, 3\), when the sum is computed along the first dimension, the output is \[5, 7, 9\]; when computed along the second dimension, the output is \[6, 15\].

**Figure 1** Example of sum computing along the first dimension
![](../figures/pypto.sum_1.png)

**Figure 2** Example of sum computing along the last dimension
![](../figures/pypto.sum_2.png)

## Prototype

```python
sum(input: Tensor, dim: int, keepdim: bool = False) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_BF16, DT_INT32, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| dim       | Input        | Dimension of reduction.<br>Any single axis is supported. |
| keepdim   | Input        | Whether to retain the reduced dimension after reduction.<br>The default value is **False**. |

## Return Value

Returns the output tensor. The shape of the output tensor depends on the **keepdim** parameter.

If the **keepdim** parameter is **True**, the reduced dimension is retained after the reduction operation. The shape of the output tensor is the same as that of the input tensor in all dimensions except the dimension specified by **dim**, where the size is **1**.

If the **keepdim** parameter is **False** (default), the reduced dimension is removed from the output tensor, while the corresponding dimension in TileShape remains unchanged. Therefore, it is recommended to reset TileShape before calling other operations.

## Constraints

1. The TileShape size must not exceed 64 KB.

2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must match that of the input shape.

Example 1: If the input shape is [m, n] and the output is [m, 1], set the TileShape to [m1, n1], where m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.sum(x, -1, True)
```

The results are as follows:

```txt
Input data x: [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
Output data y: [[6.0],
             [6.0]]
```
