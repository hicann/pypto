# pypto.SymbolicScalar.is_expression

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:38:23.098Z pushedAt=2026-08-21T07:00:27.047Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Determines whether a symbolic scalar is an expression.

## Prototype

```python
is_expression(self) -> bool
```

## Parameters

None

## Return Value

Returns **True** if it is an expression, and **False** otherwise.

## Constraints

An expression is formed by combining multiple symbols or constants through operations.

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
s3 = s2 + 5
out1 = s1.is_expression()
out2 = s2.is_expression()
out3 = s3.is_expression()
```

The result is as follows:

```python
Output data out1: False
Output data out2: False
Output data out3: True
```
