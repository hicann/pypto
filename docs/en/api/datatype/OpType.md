# OpType

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T09:05:29.899Z pushedAt=2026-08-20T12:29:39.517Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**OpType** defines the types of comparison operations, used to specify the operation type when performing comparison operations between symbolic scalars. It is mainly used in conditional judgment and logical operations.

## Prototype

```python
class OpType(enum.Enum):
     EQ = ...  # Equal operation. Determines whether two values are equal.
     NE = ...  # Not-equal operation. Determines whether two values are not equal.
     LT = ...  # Less-than operation. Determines whether the left operand is less than the right operand.
     LE = ...  # Less-than-or-equal-to operation. Determines whether the left operand is less than or equal to the right operand.
     GT = ...  # Greater-than operation. Determines whether the left operand is greater than the right operand.
     GE = ...  # Greater-than-or-equal-to operation. Determines whether the left operand is greater than or equal to the right operand.
```
