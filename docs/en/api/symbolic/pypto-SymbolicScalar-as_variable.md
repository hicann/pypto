# pypto.SymbolicScalar.as\_variable

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:35:51.509Z pushedAt=2026-08-21T06:37:51.026Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Marks the symbolic scalar as an intermediate variable.

## Prototype

```python
as_variable(self) -> None
```

## Parameters

None

## Return Value

None

## Constraints

- This is an in-place operation that modifies the internal state of the symbolic scalar.
- It is typically used to optimize expressions by marking complex expressions as intermediate variables.

## Example

```python
s = pypto.SymbolicScalar("x")
s.as_variable()  # Mark as an intermediate variable.
```
