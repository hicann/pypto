# CachePolicy

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T08:10:33.639Z pushedAt=2026-08-20T11:08:52.242Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**CachePolicy** defines the tensor cache policy, which is used to control the behavior of data in caches at various levels, optimize memory access performance, and reduce memory bandwidth consumption.

## Prototype

```python
class CachePolicy(enum.Enum):
     PREFETCH = ...        # Prefetch policy. Loads data into the cache in advance to reduce access latency.
     NONE_CACHEABLE = ...  # Non-cacheable policy. Data is not stored in the cache and is accessed directly from main memory.
```
