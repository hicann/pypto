# pypto.SymbolicScalar.concrete

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:35:53.046Z pushedAt=2026-08-21T06:57:28.795Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Obtains the concrete value of a symbolic scalar.

## Prototype

```python
concrete(self) -> int
```

## Parameters

None

## Return Value

Returns the concrete value of the symbolic scalar.

## Constraints

- This method can be called only when **is_concrete()** returns **True**.
- If the symbolic scalar is not concrete, a **ValueError** exception is thrown.

## Example

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.concrete()

# An exception is thrown if the symbolic scalar has no concrete value.
s2 = pypto.SymbolicScalar("x")
out2 = s2.concrete()
```

The result is as follows:

```python
Output data out1: 10
Output data out2: ValueError is thrown: Not concrete value
```
