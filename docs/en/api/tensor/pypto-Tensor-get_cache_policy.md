# pypto.Tensor.get\_cache\_policy

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:45:47.859Z pushedAt=2026-08-28T11:36:17.379Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains whether a cache policy is enabled.

## Prototype

```python
get_cache_policy(self, policy: CachePolicy) -> bool
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| policy  | Input      | Cache policy type. |

## Return Value

Whether the cache policy is enabled.

## Constraints

None

## Example

```python
t = pypto.tensor((16, 16), pypto.DT_FP32)
out = t.get_cache_policy(pypto.CachePolicy.PREFETCH)
```

The results are as follows:

```python
Output data out: False
```
