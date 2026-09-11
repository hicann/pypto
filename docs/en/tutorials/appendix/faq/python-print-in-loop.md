# Using the Python print Function in a Loop

<!-- md-trans-meta sourceCommit=1cc26711dfa4dcb46fee694d2efa62bd31a4f6ad translatedAt=2026-08-11T09:00:51.544Z pushedAt=2026-08-26T10:40:36.104Z -->

## Symptom

```python
@pypto.frontend.jit
def add_kernel_0(a, b, c):
    for i in pypto.loop(20):
        print("i = ", i)
        c[:] = a + b
>>>
i = 0

@pypto.frontend.jit
def add_kernel_1(a, b, c):
    for i in pypto.loop(20):
        print("i = ", i)
        if pypto.cond(i == 0):
            c[:] = a + b
        else:
            c[:] = a - b
>>>
i = 0
i = 1
```

## Possible Causes

The user operator describes the graph construction process rather than the actual execution logic. During the graph construction phase, loop execution is used only to traverse all execution paths.

- The sample code `add_kernel_0` contains only one execution path, so the loop executes only once.
- In `add_kernel_1`, there are two paths (if/else), so the loop executes twice.

## Solution

N/A
