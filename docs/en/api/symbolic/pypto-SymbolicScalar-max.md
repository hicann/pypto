# pypto.SymbolicScalar.max

<!-- md-trans-meta sourceCommit=d665ab95092c497ed3fc231aebc833bc88e9cfe6 translatedAt=2026-08-20T10:39:12.476Z pushedAt=2026-08-21T07:18:01.836Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the maximum of two symbolic scalars.

## Use Cases

Use this method when you need to compare **SymbolicScalar** values, such as dynamic dimension values obtained from `tensor.shape`.

**Difference from pypto.maximum**:

- `pypto.maximum`: Used for element-wise maximum computation on tensors.
- `SymbolicScalar.max`: Used for maximum computation on symbolic scalars.

**Python ternary expressions are not supported**:

```python
# ❌ Incorrect: Ternary expressions are not supported.
cur_seq = kv_act_seqs[b_idx]
tmp = cur_seq - s2_idx * s2_tile
actual = tmp if tmp > threshold else threshold

# ✅ Correct: Use the .max() method.
actual = (cur_seq - s2_idx * s2_tile).max(threshold)
```

## Prototype

```python
max(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar'
```

## Parameters

| Parameter Name | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input      | Symbolic scalar or integer to be compared. |

## Return Value

Returns the maximum of the two values. If both are concrete values, a constant is returned; otherwise, a symbolic expression of the SymbolicScalar type is returned.

## Constraints

- If both values are concrete, a concrete constant value is returned.
- If at least one value is not concrete, a symbolic expression is returned.

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar(5)
out1 = s1.max(s2)
out2 = s1.max(13)
s3 = pypto.SymbolicScalar("x")
out3 = s3.max(2)
```

The results are as follows:

```python
Output data out1: 10
Output data out2: 13
Output data out3: RUNTIME_Max(x, 2)
```
