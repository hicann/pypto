# pypto.relu

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:42:54.934Z pushedAt=2026-09-05T07:36:26.363Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Performs the Rectified Linear Unit (ReLU) operation on each element of **input**, retaining only the positive part and setting negative values to 0. The calculation formula is as follows:

$$
res_i = \max(0, input_i)
$$

## Prototype

```python
relu(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_INT16, DT_INT32, DT_FP16, DT_FP32, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor, whose data type and shape are the same as those of **input**.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must be the same as those of the output.

Example 1: If the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

### API Call Example

```python
input_tensor = pypto.tensor([[-2.0, 0.0, 3.0]], pypto.DT_FP32)
out = pypto.relu(input_tensor)
```

The results are as follows:

```python
Input data input: [[-2.0 0.0 3.0]]
Output data out:  [[ 0.0 0.0 3.0]]
```
