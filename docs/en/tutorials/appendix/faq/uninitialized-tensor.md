# Using an Uninitialized Tensor

<!-- md-trans-meta sourceCommit=1cc26711dfa4dcb46fee694d2efa62bd31a4f6ad translatedAt=2026-08-11T09:05:00.290Z pushedAt=2026-08-26T10:41:39.496Z -->

## Symptom

You declared a tensor using `pypto.tensor` and mistakenly assumed that, similar to `torch.empty`, it would allocate a block of uninitialized memory. Reading the tensor directly (for example, using `view`) before writing to it again causes a framework verification error or precision issue.

## Possible Causes

In PyPTO, in addition to using `pypto.full`, `pypto.zeros`, and other tensor declarations that explicitly include initialization behavior, you can also use `pypto.tensor` to declare a tensor, but this API does not include initialization behavior. In PyPTO, every tensor must be written before being read, meaning there must be a producer before a consumer. An uninitialized tensor does not allocate memory. The framework typically checks for direct use of a no-producer tensor and reports an error, but sometimes, due to a missed check, this may lead to on-device precision errors.

## Solution

Avoid using uninitialized tensors.
