# pypto.frontend.function

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:29:43.675Z pushedAt=2026-08-21T02:25:46.321Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

`pypto.frontend.function` is used to define reusable computation subgraphs or function modules, allowing the same computation logic to be reused across multiple kernel functions. This feature is intended to improve code modularity and maintainability.

Expected features:

- **Code reuse**: Encapsulate common computation patterns as functions.
- **Modular design**: Combine multiple small functions when building complex operators.
- **Type safety**: Input and output types are explicitly specified in the function signature.
- **Automatic optimization**: The compiler can inline or optimize function calls.

## Prototype

```python
@pypto.frontend.function
def reusable_function(
    arg1: pypto.Tensor,
    arg2: pypto.Tensor,
    ...
) -> pypto.Tensor:
    ...
```

## Parameters

No parameters are configured. The decorator is applied directly to the function.

## Return Value

Returns the decorated reusable function object.

## Constraints

1. Only function calls are supported; method calls are not supported.
2. Parameters: Provide the complete number of parameters (no default values) and pass them strictly in the declared order (no out-of-order or keyword arguments).
3. Recursive functions are not supported.

## Differences from pypto.frontend.jit

| Feature | pypto.frontend.jit | pypto.frontend.function |
|------|-------------------|------------------------|
| Purpose | Defines an executable kernel function | Defines a reusable sub-function |
| Compilation | Compiled into a complete computation graph | Inlined or optimized as a subgraph |
| Call | Can be directly called externally | Can only be called within a JIT function |
| Execution | Independent execution unit | Embedded into the parent function for execution |
