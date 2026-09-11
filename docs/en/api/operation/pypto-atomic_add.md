# pypto.atomic\_add

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-08-29T10:19:04.049Z pushedAt=2026-09-05T08:30:21.004Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Starting from the **dst** index position specified by **offsets**, the input tensor **src** is atomically accumulated into the corresponding region of the output tensor **dst**. Taking the 2D case as an example, the computation formula is as follows:

$$
dst\left[ offsets\left[0\right] : offsets\left[0\right] + src.shape\left[0\right],\ offsets\left[1\right] : offsets\left[1\right] + src.shape\left[1\right] \right]\ += src
$$

The same applies to other dimensions.

## Prototype

```python
atomic_add(src: Tensor, offsets: List[Union[int, SymbolicScalar]], dst: Tensor) -> None
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src     | Input      | Source operand, of type Tensor.|
| offsets | Input      | Starting offset of each dimension when **src** is written into **dst**, with elements of type int or SymbolicScalar. |
| dst     | Output      | Destination operand, of type Tensor. |

## Return Value

No return value; the result is written directly into **dst** (inplace operation).

## Constraints

1. Ensure that the valid shape of the output tensor `dst` is correct before calling `atomic_add`, as this API does not perform automatic shape inference.

2. The length of **offsets** must equal the number of dimensions of **src**/**dst**, that is, $len(offsets) == src.dim == dst.dim$; otherwise, a compilation error occurs.

3. Before calling **atomic_add**, ensure that the data in the corresponding region of **dst** is already valid; otherwise, undefined behavior may occur.

4. To ensure that the write region does not go out of bounds, for any dimension $i$, the condition $offsets[i] + src.shape[i] \le dst.shape[i]$ must be satisfied; otherwise, undefined behavior occurs.

5. When multiple cores concurrently perform **atomic_add** on overlapping regions of **dst**, the order in which the additions are executed is undefined.

## Examples

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.tensor([4, 4], pypto.DT_FP32)
pypto.atomic_add(x, [0, 0], out)
```

The results are as follows:

```txt
Input data x: [[1, 1],
           [1, 1]]
Output data out (before computing): [[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]
Output data out (after computing): [[2, 2, 0, 0],
                      [2, 2, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]
```
