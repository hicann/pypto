# pypto.amin

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-26T12:02:59.611Z pushedAt=2026-09-05T09:17:40.292Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the minimum value of a multidimensional vector along a specified dimension.

Define the specified computation dimension as the R-axis (Reduce axis) and the non-specified dimension as the A-axis (Normal axis). As shown in the figure below, for a two-dimensional matrix of shape \((2, 3)\), when the minimum value is computed along the first dimension, the output is \([1, 2, 3]\); when computed along the second dimension, the output is \([1, 4]\).

**Figure 1** Example of amin computing along the first dimension
![](../figures/pypto.amin_1.png)

**Figure 2** Example of amin computing along the last dimension
![](../figures/pypto.amin_2.png)

## Prototype

```python
amin(input: Tensor, dim: int, keepdim: bool = False) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported type: **Tensor**. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |
| dim       | Input        | Dimension on which reduction is performed.<br>Any single axis is supported. |
| keepdim   | Input        | Whether to retain the reduced dimension after reduction.<br>The default value is **False**. |

## Return Value

Returns the output tensor. The shape of the output tensor depends on the **keepdim** parameter.

If the **keepdim** parameter is **True**, the reduced dimension is retained after the reduction operation. The shape of the output tensor is the same as that of the input tensor in all dimensions except the dimension specified by **dim**, and the size of the dimension specified by **dim** is 1.

If the **keepdim** parameter is **False** (default), the reduced dimension is removed from the output tensor, while the corresponding dimension in the TileShape remains unchanged. Therefore, it is recommended to reset the
 TileShape before calling other operations.

## Constraints

1. The TileShape size must not exceed 64 KB.

2. Tensor data type description:
   - Ascend 950PR/Ascend 950DT: DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT16, DT_UINT8, and DT_INT8.
   - Atlas A3 training products/Atlas A3 inference products: DT_FP16, DT_BF16, DT_FP32, DT_INT32, and DT_INT16.
   - Atlas A2 training products/Atlas A2 inference products: DT_FP16, DT_BF16, DT_FP32, DT_INT32, and DT_INT16.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape dimensions must be consistent with those of the input.

Example 1: If the input shape is [m, n] and the output is [m, 1], set the TileShape to [m1, n1], where m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Note: If **keepdim** is set to **false**, the reduced dimension is removed from the output tensor, while the corresponding dimension in the TileShape remains unchanged. Therefore, it is recommended to reset the TileShape before calling other operations.

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.amin(x, -1, True)
```

The results are as follows:

```python
Input data x: [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
Output data y: [[1.0],
             [1.0]]
```
