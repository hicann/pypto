# pypto.set\_verify\_golden\_data

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-20T10:33:36.889Z pushedAt=2026-08-21T06:13:59.331Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

This API provides the following essential functions:

- Function 1: Sets the actual input and output lists used when the user executes an operator into the tool, so that the tool can use the same inputs for simulated computation during coarse-grained detection.
- Function 2: Sets the user's existing computation golden data into the tool, so that the tool can compare the simulated results with the golden outputs and thereby coarsely determine the correctness of the simulated results.

## Prototype

```python
set_verify_golden_data(in_out_tensors=None, goldens=None)
```

## Parameters

| Parameter | Input/Output | Description |
|-----------------|-----------|----------------------------------------------------------------------|
| **in_out_tensors** | Input | Meaning: Specifies the actual input and output lists used when the user executes the operator, and sets them into the detection tool in positional correspondence. <br> Description: This option does not need to be set in JIT compilation mode.  <br> Type: List[Union(pypto.Tensor, torch.Tensor)] <br> Value Range: N/A <br> Default Value: N/A |
| **goldens** | Input | Meaning: Sets the user's existing computation golden data outputs into the tool for comparison detection. <br> Description: This list has the same length and corresponding positions as the operator's input and output parameter lists. Its elements correspond to the operator's output parameters, and the positions of input parameters should be set to **None**. If a corresponding position is set to **None**, data comparison at that position is skipped. <br> Type: List[Union(pypto.Tensor, torch.Tensor)] <br> * The device attribute of **torch.Tensor** must be CPU; NPU is not supported. <br> Value Range: N/A <br> Default Value: N/A |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

This function takes effect only after **pypto.set_verify_options(enable_pass_verify=True)** is set.

## Example

```python
set_verify_golden_data(goldens=[None, None, golden_out0])
set_verify_golden_data([real_in0, real_in1, real_out0], [None, None, golden_out0])
```
