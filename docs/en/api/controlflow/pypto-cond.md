# pypto.cond

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T08:06:24.603Z pushedAt=2026-08-20T10:49:12.536Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Defines an if conditional operation to implement the if functionality in Python.

## Prototype

```python
cond(scalar: SymInt) -> pypto_impl.RecordIfBranch
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| scalar | Input | Conditional expression, which can be an integer or a SymbolicScalar, used to determine whether the condition is true. |

## Return Value

**pypto\_impl.RecordIfBranch**: Returns a conditional branch object for use in Python if statements.

## Constraints

- Must be used together with Python if, elif, and else statements.
- The conditional expression is recorded into the computation graph.
- Nested conditional statements are supported.
- When the function is not decorated with **@pypto.frontend.jit** or **@pypto.frontend.function**, the conditional expression must be wrapped with **pypto.cond**.

## Example

```python
# Without a decorator, wrap the conditional expression with pypto.cond.
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, power_of_2(max_unroll_times), name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.cond(pypto.is_loop_end(s2_idx, bn_per_batch)):
            ...

# With a decorator, no pypto.cond wrapping is required.
@pypto.frontend.jit
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, power_of_2(max_unroll_times), name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.is_loop_end(s2_idx, bn_per_batch):
            ...
```
