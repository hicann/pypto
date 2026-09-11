# pypto.gathermask

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:05:01.879Z pushedAt=2026-09-05T07:36:26.333Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Based on the built-in mask selected by **PatternMode**, elements at positions where the corresponding bit of the input tensor is 1 are collected to form the output tensor, while values at positions where the bit is 0 are discarded. **PatternMode** has seven modes:

- **PatternMode** = 1: For the last axis, take the first element of every two elements.
- **PatternMode** = 2: For the last axis, take the second element of every two elements.
- **PatternMode** = 3: For the last axis, take the first element of every four elements.
- **PatternMode** = 4: For the last axis, take the second element of every four elements.
- **PatternMode** = 5: For the last axis, take the third element of every four elements.
- **PatternMode** = 6: For the last axis, take the fourth element of every four elements.
- **PatternMode** = 7: For the last axis, take all elements.

## Prototype

```python
gathermask(self: Tensor, pattern_mode: int) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| self   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |
| pattern_mode | Input      | Source operand.<br>Type: int, with a value range of 1 to 7. |

## Return Value

Returns the output tensor. The data type of the output tensor is the same as that of **self**, and the shape of the output tensor is as follows:

- When **pattern_mode** <= 2, the last axis of the output shape is half of the last axis of **self.shape**, and the other axes are the same as those of the input shape.
- When 2 < **pattern_mode** < 7, the last axis of the output shape is one quarter of the last axis of **self.shape**, and the other axes are the same as those of the input shape.
- When **pattern_mode** = 7, the output shape is the same as the input shape.

## Constraints

1. When 1 <= **pattern_mode** <= 2:
   - The last axis of **self.shape** must be divisible by 2.
   - The last axis of **tileshape** must be a multiple of 2.
   - The last axis of **viewshape** must be a multiple of 2.
   - The last axis of **self.shape** is not tiled by view.
2. When 3 <= **pattern_mode** <= 6:
   - The last axis of **self.shape** must be a multiple of 4.
   - The last axis of **tileshape** must be a multiple of 4.
   - The last axis of **viewshape** must be a multiple of 4.
   - The last axis of **self.shape** is not tiled by view.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must be the same as those of the output.

If the input **self** is [x, y, z], **pattern_mode** is 1, and the output is [x, y, z/2], with the TileShape set to [x1, y1, 2*z1], then x1, y1, and 2\*z1 are used to tile the x, y, and z axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([3, 6], pypto.DT_INT32)        # shape (3, 6)
pattern_mode = 1
y = pypto.gathermask(x, pattern_mode)
```

The results are as follows:

```python
Input data x: [[0,  1,  2,  3,  4,  5],
             [6,  7,  8,  9,  10,  11],
             [12,  13,  14,  15,  16,  17]]
     pattern_mode: 1
Output data y: [[0,  2,  4],
             [6,  8,  10],
             [12,  14,  16]]
```
