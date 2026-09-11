# pypto.SymbolicScalar Constructor

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:33:35.772Z pushedAt=2026-08-21T06:16:46.855Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Creates a new **SymbolicScalar** instance, supporting multiple construction methods.

## Prototype

```python
__init__(self,
         arg0: Union[int, str, 'SymbolicScalar'] = None,
         arg1: Union[int, None] = None
) -> None
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| self      | Output       | Reference to the instance object, automatically passed by Python. |
| arg0      | Input        | Value or name of the SymbolicScalar, which can be:<br> - **int**: Integer value, creating a constant SymbolicScalar.<br> - **str**: Symbol name, creating a SymbolicScalar.<br> - **SymbolicScalar**: Another SymbolicScalar, used for copying. |
| arg1      | Input        | Value of the SymbolicScalar, optionally used only when **arg0** is a string. |

## Return Value

None

## Constraints

- If **arg0** is an integer, arg1 is ignored.
- If **arg0** is a string and **arg1** is an integer, a SymbolicScalar with an initial value is created.
- If **arg0** is a string and **arg1** is **None**, a SymbolicScalar without an initial value is created.
- If **arg0** is a SymbolicScalar, its underlying implementation object is copied.

## Example

```python
a = pypto.SymbolicScalar()
b = pypto.SymbolicScalar(10)
c = pypto.SymbolicScalar("x")
d = pypto.SymbolicScalar("x", 10)
e = pypto.SymbolicScalar(c)
```
