# pypto.normal

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:33:04.649Z pushedAt=2026-09-05T07:36:26.354Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Not supported
- Atlas A2 training products/Atlas A2 inference products: Not Supported

## Description

Generates random numbers following a normal (Gaussian) distribution with the specified shape, where each element follows a distribution with a mean of 0 and a variance of 1.
$$
x_i \sim N(0, 1)
$$

## Prototype

```python
normal(shape: List[int], key: List[int], counter: List[int], alg: List[int], dtype: DataType) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|--------------------------------------------------------------------|
| **shape**   | Input      | Shape of the output Tensor.<br>The length supports 1 to 4 dimensions.                                        |
| **key**     | Input      | Seed of the random number generator.<br>The length only supports 1.                                         |
| **counter** | Input      | Counter of the random number generator.<br>The length only supports 2.                                          |
| **alg**     | Input      | Random number generation algorithm. Currently only value 1 (Philox algorithm) and 3 (**auto_select**, which selects the Philox algorithm) are supported.<br>The length only supports 1. |
| **dtype**   | Input      | Data type of the output tensor.<br>Supported data types: DT_FP32, DT_FP16, and DT_BF16.            |

## Constraints

- Splitting shape into multiple view shapes is not supported. The view shape must be consistent with the input shape.
- Splitting shape into multiple tile shapes is not supported. The tile shape must be consistent with the input shape.
- The last axis of the tile shape must be a multiple of 4.
- `counter[0]` is hardcoded to 0 internally. Although the API accepts a counter list of length 2, the value of `counter[0]` is ignored, and the Philox counter actually used is `[0, counter[1]]`.

## Return Value

Returns a tensor with the specified shape and data type dtype, whose elements follow a normal distribution with a mean of 0 and a variance of 1.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same dimension count and shape as the input.

For example, if the input shape is `[m, n]` and the output shape is `[m, n]`, then `TileShape` should be set to `[m, n]`.

```python
pypto.set_vec_tile_shapes(4, 4)
```

### API Call Example

```python
shape = [4, 4]
key = [1234]
counter = [0, 1]
alg = [1]
dtype = pypto.DT_FP32

y = pypto.normal(shape, key, counter, alg, dtype)
```

The results are as follows:

```python
Output data y: [[-0.32364845  1.8577391   0.39556974  0.2311697 ]
            [ 0.24243996 -1.9485782  -0.12983137  2.7137496 ]
            [ 1.6558666   2.0938187  -0.90338254  0.8765667 ]
            [ 0.86518306  0.01034508  0.2893259   0.01748212]]
```
