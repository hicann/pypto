# pypto.loop_unroll

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T08:07:16.213Z pushedAt=2026-08-24T03:09:08.648Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

**pypto.loop_unroll** is a loop iterator function that supports loop unrolling. It is similar to **pypto.loop** but adds the **unroll_list** parameter to support multiple unrolling methods.

## Prototype

```python
loop_unroll(start: SymInt = 0, stop: SymInt, step: SymInt = 1, *, name: str = None, idx_name: str = None, unroll_list: List[int] = None, submit_before_loop: bool = False) -> Iterator[Tuple[SymInt, int]]
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| *args | Input | Three optional parameters: the loop start value (**start**), the loop stop value (**stop**), and the loop step (**step**). The following three forms are supported:<br> - Single-parameter form: **stop** (**SymInt**), with the start value defaulting to **0** and the step defaulting to **1**. Equivalent to: **loop_unroll(0, stop, 1)**.<br> - Two-parameter form: **start** (**SymInt**) and **stop** (**SymInt**). Equivalent to **loop_unroll(start, stop, 1)**.<br> - Three-parameter form: **start** (**SymInt**), **stop** (**SymInt**), and **step** (**SymInt**). Equivalent to **loop_unroll(start, stop, step)**. |
| **kwargs | Input | - **name (str)**: Loop identifier name, defaulting to **f"loop_{loop_idx}"**.<br> - **idx_name (str)**: Name of the loop index variable, defaulting to **f"loop_idx_{loop_idx}"**.<br> - **unroll_list (List[int])**: Set of loop levels to be unrolled, defaulting to an empty set. The loop provides the unrolling methods defined in **unroll_list**. When the unrolling count is n, the loop step becomes **step*n**, and each iteration executes the loop body n times. Each unrolling count generates a different code path.<br> - **submit_before_loop (bool)**: Whether to submit computation before the loop starts, defaulting to **False**. When enabled, the currently accumulated computation tasks are forcibly submitted to the AI Core for execution before the loop starts. |

## Return Value

Returns an iterator that yields a tuple **(idx, unroll_factor)** per iteration, where **idx** is the current loop index value and **unroll_factor** identifies the currently selected unrolling method.

## Constraints

- The unrolling factor list is sorted, deduplicated, and always includes 1.
- Unrolling factors are sorted in descending order.
- Each unrolling factor generates a sub-loop.
- Using **loop_unroll** with **unroll_list** configured across multiple nested loops significantly increases the number of compiled graphs, which may impact compilation performance.

## Example

```python
for idx, unroll_factor in pypto.loop_unroll(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx", unroll_list=[1, 2, 4]):
   ...
```
