# pypto.function

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T08:06:37.642Z pushedAt=2026-08-24T03:05:07.814Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Defines a PyPTO computation function. Operations required for building a computation graph can be added within this function.

## Prototype

```python
function(name: str, *args, **kwargs) -> Iterator
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| name   | Input      | Name of the function, used to identify the computation graph. |
| *args  | Input      | The input tensor is obtained from these arguments. |

## Return Value

Returns a context manager, which is used in a **with** statement.

## Constraints

None

## Example

```python
with pypto.function("main", a, b, c):
    pypto.set_vec_tile_shapes(16, 16)
    for _ in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        c[:] = a + b
```
