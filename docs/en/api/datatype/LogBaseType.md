# LogBaseType

<!-- md-trans-meta sourceCommit=d665ab95092c497ed3fc231aebc833bc88e9cfe6 translatedAt=2026-08-20T09:05:31.147Z pushedAt=2026-08-24T03:15:05.072Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**LogBaseType** defines the base type for logarithmic operations, used to specify the base for logarithmic functions. It supports common logarithmic operations including natural logarithm, base-2 logarithm, and base-10 logarithm.

## Prototype

```python
class LogBaseType(enum.Enum):
     LOG_E = ...   # Natural logarithm base; logarithmic operation with base approximately 2.718..
     LOG_2 = ...   # Base-2 logarithmic operation, commonly used in information theory and computer science.
     LOG_10 = ...  # Base-10 logarithmic operation, commonly used in scientific computing and engineering applications.
```
