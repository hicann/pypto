# pypto.pass\_verify\_print

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:29:45.443Z pushedAt=2026-08-21T02:30:43.473Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

When the Verify feature for precision debugging is enabled, this API saves the computation result of the specified tensor to a data file.

## Prototype

```python
pass_verify_print(*values, cond: Union[int, SymbolicScalar] = 1) -> None
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| *values | Input      | Meaning: Specifies the data or information to print. <br> Description:<br> **pypto.Tensor**: Prints the tensor data; int/**pypto.SymbolicScalar**: Prints the corresponding value; other Python objects: Prints the corresponding string representation. <br> Type: List[pypto.Tensor,int,pypto.SymbolicScalar,Object] <br> Value range: N/A <br> Default value: N/A |
| cond    | Input      | Meaning: Specifies the condition for printing the data. <br> Description: If the expression evaluates to 1, the specified data is printed; if it evaluates to 0, the data is not printed. This parameter can be omitted, in which case the default value is used. Explicitly passing **None** is not supported. <br> Type: Union[int,pypto.SymbolicScalar] <br> Value range: 0,1 <br> Default value: 1 |

## Return Value

None

## Constraints

This function takes effect only after **pypto.set_verify_options(enable_pass_verify=True)** is set.

## Example

```python
verify_options = {
        "enable_pass_verify": True,
      }

@pypto.frontend.jit(verify_options=verify_options)
def user_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor):
    ...
    for idx in pypto.loop(10):
        t0 = pypto.tensor(...)
        t1 = pypto.tensor(...)
        t2 = pypto.SOME_OP1(t0, t1)
        pypto.pass_verify_print(t2)
        t3 = pypto.SOME_OP2(t0, t2)
        pypto.pass_verify_print(t3, cond=(idx == 5))
         ...
```
