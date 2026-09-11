# pypto.Tensor.set_cache_policy

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:55:21.811Z pushedAt=2026-08-28T11:36:17.397Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Sets the attributes of the tensor cache.

## Prototype

```python
set_cache_policy(self, policy: CachePolicy, value: bool) -> None
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| policy  | Input      | Cache policy type. Available values: <br> - **CachePolicy.NONE_CACHEABLE**: The chip provides L2 Cache capability, but for some operators, having L2 Cache is worse than not having it due to their characteristics. After configuration, this tensor does not pass through L2 Cache. Common scenarios include: <br>   - For constants such as **weight**, if the operator only reads from **out** once and does not reuse it, there is no need to enter L2. <br>   - If the output shape is too large and the memory first used by the lower-level operator is not the final output of the upper-level operator, entering L2 triggers the lower-level operator to write back to output, degrading performance. |
| value   | Input      | Whether to enable this cache policy. |

## Return Value

None

## Constraints

None

## Example

```python
t = pypto.tensor((16, 16), pypto.DT_FP32)
t.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
```
