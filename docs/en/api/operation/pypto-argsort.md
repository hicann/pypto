# pypto.argsort

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:15:17.778Z pushedAt=2026-09-05T08:30:07.548Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the indices of the input after sorting along the specified axis in ascending or descending order.

## Prototype

```python
argsort(input: Tensor, dim: Optional[int]=None, descending: bool=True) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of Tensor: DT_FP32 and DT_FP16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| dim     | Input      | Dimension along which to sort.<br>Supports axes 1 to 4.|
| descending | Input      | If **True**, returns the indices in descending order. If **False**, returns the indices in ascending order. |

## Return Value

Returns a **Tensor** containing the indices of the input after sorting along the **dim** axis according to **descending**.

## Constraints

1. Currently, splitting **ViewShape** along the **dim** axis is not supported, which means **ViewShape[dim]** must equal **InputShape[dim]**.
2. Currently, **TileShape** along the **dim** axis must be a multiple of 32, which means **TileShape[dim]** % 32 = 0.
3. When the shape is large $(\frac{TileShape\ Size}{TileShape[dim]} * CeilAlign(ViewShape[dim], 32) >= 6144)$, the number of tile splits along the sorting axis must be less than 128.
4. For four-dimensional input, sorting along axis 0 is not supported.
5. When equal values are encountered during sorting, stable sorting is used to return the corresponding indices.
6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must match that of the input shape.

For example, if the input shape is [m, n, p], **dim** is 2, and **descending** is **True**, the output shape is [m, n, p]. If **TileShape** is set to [m1, n1, p1], then m1, n1, and p1 are used to split the m, n, and p axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.argsort(x, -1, True)
```

The results are as follows:

```python
Input data x: [[1.0 2.0 3.0],
            [1.0 2.0 3.0]]
Output data y: [[2, 1, 0],
            [2, 1, 0]]
```
