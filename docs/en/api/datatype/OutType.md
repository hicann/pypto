# OutType

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T09:05:31.457Z pushedAt=2026-08-20T12:29:57.818Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**OutType** defines the type of output data. It specifies the output format of certain operations (such as comparison operations), distinguishing between Boolean value output and bit value output.

## Prototype

```python
class OutType(enum.Enum):
     BOOL = ...  # Boolean output type, which outputs True or False.
     BIT = ...   # Bit output type, which outputs a bit value of 0 or 1.
```
