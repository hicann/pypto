# pypto.is\_loop\_end

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T08:07:05.308Z pushedAt=2026-08-20T10:54:13.983Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Determines whether the current iteration is the end of the loop.

## Prototype

```python
def is_loop_end(scalar: SymInt) -> SymbolicScalar
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| scalar  | Input      | Index of the current loop. |

## Return Value

Returns a symbolic scalar expression indicating whether it is the end of the loop (a Boolean value).

## Constraints

- **scalar** must be a symbolic scalar returned by the loop iterator.
- If it is not a loop index, a **ValueError** exception is thrown.
- When the function is not decorated with **@pypto.frontend.jit** or **@pypto.frontend.function**, the conditional expression must be wrapped with **pypto.cond**.

## Example

```python
# No decorator is used, so the conditional expression must be wrapped with pypto.cond.
def kernel():
    ...
    for idx in pypto.loop(0, 10, 1):
        if pypto.cond(pypto.is_loop_end(idx)):
            ...

# A decorator is used, so wrapping with pypto.cond is not required.
@pypto.frontend.jit
def kernel():
    ...
    for idx in pypto.loop(0, 10, 1):
        if pypto.is_loop_end(idx):
            ...
```
