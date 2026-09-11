# pypto.topk

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:59:50.491Z pushedAt=2026-09-05T07:36:26.378Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the top k largest or smallest values along the last dimension and their corresponding indices.

If the input is a vector, this operation finds the top k largest or smallest values and their corresponding indices in the vector. If the input is a matrix, it computes the top k largest or smallest values and their corresponding indices in each row along the last dimension. As shown in the following figure, a two-dimensional matrix with shape \(4, 32\) is sorted with k set to 1, and the output is \[\[32\] \[32\] \[32\] \[32\]\].

![](../figures/nz-reduce.png)

## Prototype

```python
topk(input: Tensor, k: int, dim: Optional[int] = None, largest: bool = True, algo: TopKAlgo = TopKAlgo.MERGE_SORT) -> Tuple[Tensor, Tensor]
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor:<br>- MERGE_SORT: DT_FP32.<br>- RADIX_SELECT: DT_BF16, DT_FP16, and DT_FP32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| k       | Input      | Number of elements to return.<br>The value of k must satisfy: 1 <= k <= input.shape[dim]. |
| dim     | Input      | Dimension along which to sort.<br>Currently, only sorting along the last dimension is supported, that is, dim = -1 or dim = input.shape.size() - 1. |
| largest | Input      | If **True**, returns the largest elements. If **False**, returns the smallest elements. |
| algo    | Input      | Algorithm enumeration type that controls the TopK computation flow. For details, see [TopKAlgo](../datatype/TopKAlgo.md).<br>Defaults to **MERGE_SORT** (merge sort algorithm). |

## Return Value

Returns a named tuple (values, indices) containing the values and indices of the k largest or smallest elements in each row of input along the specified dimension dim.

## Constraints

1. Only the topk operation on the last axis is supported.
2. When the MERGE_SORT algorithm is selected, the last axis of TileShape must be less than 22KB\(TileShape\[-1\]\*4 < 22KB\).
3. When the RADIX_SELECT algorithm is selected, let the last axis of TileShape be tile. Then a temporary space of 2\*tile\*sizeof\(srcType\)+6\*tile+1024+max\(1024, 8\*tile\) is required, and the temporary space plus the input and output tile blocks must not exceed the UB size.
4. k <= TileShape\[-1\] && k <= input.shape\[-1\].
5. The **RADIX_SELECT** algorithm is supported only on Ascend 950PR/Ascend 950DT.
6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must match that of the input shape.

Example 1: If the input shape is [m, n, p], **dim** is 2, and **largest** is **True**, the output is [m, n, k]. When **TileShape** is set to [m1, n1, p1], m1, n1, and p1 are used to tile the m, n, and p axes, respectively. p1 must be greater than or equal to k. The k axis does not support tiling and must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.MERGE_SORT)
```

The results are as follows:

```python
Input data x: [[1.0 2.0 3.0],
            [1.0 2.0 3.0]]
Output data y[0]: [[3.0 2.0],
               [3.0 2.0]]
Output data y[1]: [[2, 1],
               [2, 1]]
```
