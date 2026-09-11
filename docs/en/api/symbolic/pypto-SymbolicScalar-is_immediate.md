# pypto.SymbolicScalar.is_immediate

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:38:20.583Z pushedAt=2026-08-21T07:02:31.010Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Determines whether a symbolic scalar is an immediate value (a concrete numeric value).

## Prototype

```python
is_immediate(self) -> bool
```

## Parameters

None

## Return Value

Returns **True** if it is an immediate value, and **False** otherwise.

## Constraints

An immediate value is a constant whose concrete numeric value can be determined at compile time.

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_immediate()
out2 = s2.is_immediate()
```

The result is as follows:

```python
Output data out1: True
Output data out2: False
```
