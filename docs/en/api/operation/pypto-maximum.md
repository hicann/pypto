# pypto.maximum

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:27:36.805Z pushedAt=2026-09-05T07:36:26.349Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the element-wise maximum of the input and another input. Supports 2D, 3D, or 4D tensors.

## Precautions

- **SymbolicScalar parameters are not supported**: To compare **SymbolicScalar** values, use the [SymbolicScalar.max()](../symbolic/pypto-SymbolicScalar-max.md) method.
- At least one of the two parameters must be of the Tensor type.

## Prototype

```python
maximum(
    input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]
) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| input | Input | Source operand.<br>Supported types are int, float, Element, and Tensor.<br>When the type is int or float, it is automatically converted to the Element type, where int corresponds to DT_INT32 and float corresponds to DT_FP32. To use other data types, construct them through Element.<br>The Tensor and Element data types supported vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other | Input | Source operand.<br>Supported types are int, float, Element, and Tensor.<br>When the type is int or float, it is automatically converted to the Element type, where int corresponds to DT_INT32 and float corresponds to DT_FP32. To use other data types, construct them through Element.<br>The Tensor and Element data types supported vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**).<br>The type and data type must be the same as those of the source operand. |

At least one of Source Operand 1 and Source Operand 2 must be a tensor.

## Return Value

When both source operands are tensors, the two tensors must satisfy the broadcast relationship. This API returns a tensor with the same shape as the broadcast result of Source Operand 1 and Source Operand 2, with the same data type as the source operands, whose elements are the element-wise maximum of Source Operand 1 and Source Operand 2. When the source operands are tensors, both Source Operand 1 and Source Operand 2 support multi-axis broadcasting.

When one of the two source operands is a tensor, a tensor with the same shape as the input tensor is returned, whose elements are the element-wise maximum of Source Operand 1 and Source Operand 2.

## Constraints

1. When both inputs are of the Tensor type, the supported data types are as follows:
   - Ascend 950PR/Ascend 950DT: DT_INT32, DT_UINT32, DT_FP32, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_UINT8, and DT_INT8.
   - Atlas A3 training products/Atlas A3 inference products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16.
   - Atlas A2 training products/Atlas A2 inference products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16.
2. When one input is of the Tensor type and the other is of the Element type, the supported data types are as follows:
   - Ascend 950PR/Ascend 950DT: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16.
   - Atlas A3 training products/Atlas A3 inference products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16.
   - Atlas A2 training products/Atlas A2 inference products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape dimensions must be consistent with the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([3], pypto.DT_INT32)
b = pypto.tensor([3], pypto.DT_INT32)
out = pypto.maximum(a, b)
```

The results are as follows:

```python
Input data a: [0, 2, 4]
Input data b: [3, 1, 3]
Output data out: [3, 2, 4]
```
