# TopKAlgo

<!-- md-trans-meta sourceCommit=ef159c53aaffca68734dbf8caedc52ff7e051796 translatedAt=2026-08-20T10:23:00.616Z pushedAt=2026-08-20T12:54:23.224Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**TopKAlgo** defines the TopK algorithm and controls how TopK computation is processed.

## Prototype

```python
class TopKAlgo(enum.Enum):
     MERGE_SORT = ...    # Merge sort algorithm.
     RADIX_SELECT = ...  # Radix selection algorithm.
```

## Parameters

| Value | Description |
|:-------|:-----|
| MERGE_SORT | Merge sort algorithm. Sorts the entire tensor and then selects the first *k* elements. |
| RADIX_SELECT | Radix selection algorithm. First finds the *k*-th element, and then selects the first *k* elements based on it. |

## Usage Recommendations

1. **Default behavior**: If no algorithm is specified, the `MERGE_SORT` mode is used by default.
2. **Performance-sensitive scenarios**: The `RADIX_SELECT` mode is recommended, with a time complexity of O(n).
3. **Scenarios with low performance requirements**: The `MERGE_SORT` mode can be used, with a time complexity of O(n log n).

## Example

```python
import pypto

# Create a tensor.
x = pypto.tensor([2, 3], pypto.DT_FP32)

# Use the merge sort algorithm.
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.MERGE_SORT)

# Use the radix select algorithm.
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.RADIX_SELECT)

# Use the merge sort algorithm by default.
y = pypto.topk(x, 2, -1, True)
```
