# pypto.assemble

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:15:17.709Z pushedAt=2026-09-05T08:30:13.233Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Assigns the input tensor `input` to the corresponding region of the output tensor `out`, based on the output index position specified by `offsets`.

## Prototype

```python
assemble(input: Tensor, offsets: List[Union[int, SymbolicScalar]], out: Tensor) -> None

assemble(inputs: List[Tuple[Tensor, List[Union[int, SymbolicScalar]]]], out: Tensor, parallel: bool = False) -> None
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported data types: data types supported by PyPto.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| inputs   | Input      | Tuple list consisting of source operands and output offsets.<br>Supported data type for each element: data types supported by PyPto.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| offsets | Input      | Offset relative to the target output.<br>Ensure that offsets is smaller than the shape of `out`.          |
| out     | Output      | Destination operand, which must have the same number of dimensions as `input`.<br>Supported data types: data types supported by PyPto.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| parallel | Input      | Whether to execute in parallel.<br>Default value: False. |

## Return Value

No return value; **out** is modified directly.

## Constraints

1. Ensure that the valid shape of the output tensor `out` is correct before calling `assemble`, as this API does not perform automatic shape inference.
2. The input tensor `input` and the output tensor `out` must have the same number of dimensions.

## Examples

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.tensor([4, 4], pypto.DT_FP32)
offsets = [0, 0]
pypto.assemble(x, offsets, out)

y = pypto.tensor([2, 2], pypto.DT_FP32)
pypto.assemble([(x, offsets), (y, [2, 2])], out)
```

The results are as follows:

```python
Output data x: [[1, 1]
           [1, 1]]
Input data out: [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
Output data out: [[1, 1, 0, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
Output data out1: [[1, 1, 0, 0],
              [1, 1, 0, 0],
              [0, 0, 1, 1],
              [0, 0, 1, 1]]
```
