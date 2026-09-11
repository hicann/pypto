# pypto.SymbolicScalar.min

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:39:45.629Z pushedAt=2026-08-21T07:21:33.491Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Computes the minimum of two symbolic scalars.

## Use Cases

Use this method when you need to compare **SymbolicScalar** values, such as dynamic dimension values obtained from `tensor.shape`.

**Difference from pypto.minimum**:

- `pypto.minimum`: Used for element-wise minimum computation on tensors.
- `SymbolicScalar.min`: Used for minimum computation on symbolic scalars.

**Python ternary expressions are not supported**:

```python
# ❌ Error: Ternary expressions are not supported.
cur_seq = kv_act_seqs[b_idx]
tmp = cur_seq - s2_idx * s2_tile
actual = tmp if tmp < s2_tile else s2_tile

# ✅ Correct: Use the .min() method.
actual = (cur_seq - s2_idx * s2_tile).min(s2_tile)
```

## Prototype

```python
min(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar'
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input      | Symbolic scalar or integer to be compared. |

## Return Value

Returns the minimum of the two values.

## Constraints

- If both values are concrete, a concrete constant value is returned.
- If at least one value is not concrete, a symbolic expression is returned.

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar(5)
out1 = s1.min(s2)
out2 = s1.min(3)
s3 = pypto.SymbolicScalar(x)
out3 = s3.min(2)
```

The results are as follows:

```python
Output data out1: 5
Output data out2: 3
Output data out3: RUNTIME_Min(x, 2)
```
