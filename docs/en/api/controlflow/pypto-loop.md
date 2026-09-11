# pypto.loop

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T08:10:33.358Z pushedAt=2026-08-24T03:10:03.004Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Defines a loop operation that implements the functionality of a **for** loop in Python.

## Prototype

```python
loop(stop: SymInt, /, **kwargs) -> Iterator[SymInt]
loop(start: SymInt, stop: SymInt, step: Optional[SymInt] = 1, /, **kwargs) -> Iterator[SymInt]
```

## Parameters

| Parameter | Input/Output | Description |
|-------------------|-----------|----------------------------------------------------------------------|
| start | Input | Start value of the loop. |
| stop | Input | End value of the loop. |
| step | Input | Step of each iteration. |
| **kwargs | Input | - **name (str)**: Loop identifier name, defaulting to **f"loop_{loop_idx}"**.<br> - **idx_name (str)**: Name of the loop index variable, defaulting to **f"loop_idx_{loop_idx}"**.<br> - **submit_before_loop (bool)**: Whether to submit computation before the loop starts, defaulting to **False**. When enabled, the currently accumulated computation tasks are forcibly submitted to the AI Core for execution before the loop starts.<br> - **parallel (bool)**: Whether to mark the loop as parallel-schedulable, defaulting to **False**. When set to **True**, it indicates that there is no dependency between the iterations of the loop, and they can be scheduled for parallel execution. This must be used together with the **device_sched_parallelism** configuration item. |

## Return Value

Returns a generator that yields a symbolic integer representing the value of each iteration in sequence.

## Constraints

The loop index variable returned by **loop** (such as **b_idx**) is of the **SymInt** type and cannot be used as a list subscript index.

## Example

```python
for _ in pypto.loop(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
   ...
```
