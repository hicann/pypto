# pypto.SymbolicScalar.is\_symbol

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:38:50.721Z pushedAt=2026-08-21T07:08:25.092Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Determines whether a **SymbolicScalar** is symbolic.

## Prototype

```python
is_symbol(self) -> bool
```

## Parameters

None

## Return Value

Returns **True** if it is symbolic, and **False** otherwise.

## Constraints

A symbolic variable is a variable whose concrete value cannot be determined at runtime and is used to construct a computation graph.

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_symbol()
out2 = s2.is_symbol()
```

The result is as follows:

```python
Output data out1: False
Output data out2: True
```
