# pypto.one\_hot

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:33:48.148Z pushedAt=2026-09-05T07:36:26.354Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Converts an integer tensor into the corresponding one-hot encoding, where each integer is converted into a vector with only the corresponding position set to 1 and all other positions set to 0.

## Prototype

```python
one_hot(input: Tensor, num_classes: int) -> Tensor
```

## Parameters

| Parameter   | Input/Output | Description                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| **input**       | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_INT8, DT_INT16, DT_INT32, and DT_INT64.<br>Supports 1 to 3 dimensions.<br>Internal elements must be non-negative.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| **num_classes** | Input      | Length of the one-hot encoding.<br>Must be greater than the maximum element in **input**. |

## Return Value

Returns a tensor with a shape of \(input, num\_classes\) and a data type of DT\_INT64.

## Constraints

`TileShape` is used to tile the output. Its dimensions must match those of the output, and its last axis must equal `num_classes`.

Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape dimensions must be consistent with the output.

Example 1: If the input shape is [m, n], the output is [m, n, t], where t = num_classes. If TileShape is set to [m1, n1, t1], then m1 and n1 are used to tile the m and n axes, respectively. t1 must be equal to `num_classes`. The t axis cannot be tiled and must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([3], pypto.DT_INT32)
y = pypto.one_hot(x, 5)
```

The results are as follows:

```python
Input data x: [0, 2, 4]
Output data y: [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]
```
