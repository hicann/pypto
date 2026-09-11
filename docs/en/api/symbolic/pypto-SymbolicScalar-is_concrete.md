# pypto.SymbolicScalar.is\_concrete

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:36:34.926Z pushedAt=2026-08-21T06:58:15.035Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Determines whether a symbolic scalar has a concrete value.

## Prototype

```python
is_concrete(self) -> bool
```

## Parameters

None

## Return Value

Returns **True** if it has a concrete value; otherwise, returns **False**.

## Constraints

- Constant values are always concrete.
- Certain expressions may also have concrete values under specific conditions.

## Example

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.is_concrete()
s2 = pypto.SymbolicScalar("x")
out2 = s2.is_concrete()
```

The result is as follows:

```python
Output data out1: True
Output data out2: False
```
