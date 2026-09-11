# Precision Issue Occurs When valid_shape Is Not Passed to view

<!-- md-trans-meta sourceCommit=1cc26711dfa4dcb46fee694d2efa62bd31a4f6ad translatedAt=2026-08-11T09:06:00.843Z pushedAt=2026-08-26T10:45:42.431Z -->

## Symptom

In some scenarios, precision issues occur when `valid_shape` is not passed to `view`.

## Possible Causes

When the input tensor of the `view` API does not have a correct validShape, the output validShape cannot be correctly inferred by the framework.

## Solution

If you suspect that the validShape inference for `view` is incorrect, first pass a `valid_shape` to `view` and check whether the output meets expectations.

A typical scenario where `valid_shape` must be passed:

When the input validShape depends on another tensor identifier, `dynValidShape` must be passed. In the following scenario, the validShape `curSeq` of `q0` comes from another tensor and cannot be obtained through inference:

```python
# Input: input [B, S, H]
# Input: act_seqs [B]
# Output: out [B, S, H]
# Computation process: AddS
# The code is as follows:
for b_idx in pypto.loop(B, name="b_loop", idx_name="b"):
    cur_seq = act_seqs[b_idx]
    a0 = pypto.view(input, [1, S, H], [b_idx, 0, 0], valid_shape=[1, cur_seq, H])
    a1 = a0 + 1.0
    out[b_idx:, :, :] = a1
```
