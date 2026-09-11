# pypto.set\_verify\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:08:46.742Z pushedAt=2026-08-26T09:10:38.135Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Sets the self-check switch of the precision debugging Verify feature and the functional options corresponding to the feature.

## Prototype

```python
set_verify_options(*,
                   enable_pass_verify: Optional[bool] = None,
                   pass_verify_save_tensor: Optional[bool] = None,
                   pass_verify_save_tensor_dir: Optional[str] = None,
                   pass_verify_pass_filter: Optional[List[str]] = None,
                   pass_verify_error_tol: Optional[List[float]] = None,
                   ) -> None
```

## Parameters

| Parameter                       | Input/Output | Description                                                                 |
|---------------------------------|-----------|----------------------------------------------------------------------|
| **enable_pass_verify**              | Input      | Meaning: Overall enable switch that determines whether all *pass_verify_* options and APIs take effect. <br> Description: **True** (enabled). <br> Type: bool <br> Value Range: True/False <br> Default Value: False |
| **pass_verify_save_tensor**         | Input      | Meaning: Whether to save the simulated computation data to disk. <br> Description: **True** (save to disk). <br> Type: bool <br> Value Range: True/False <br> Default Value: False |
| **pass_verify_save_tensor_dir**     | Input      | Meaning: Save path for the check results and data. <br> Description: A string specifying an absolute path. <br> Type: str <br> Default Value: {RUNNING_DIR}/output/output_{TS} |
| **pass_verify_pass_filter**         | Input      | Meaning: List of Pass names to be self-checked. <br> Description: Valid Pass names. If not specified, the following passes are checked by default: ["ExpandFunction", "ProcessAtomic", "L1CopyInReuseMerge", "InferDynShape", "PreGraphProcess", "InferParamIndex", "CodegenPreproc"]. If **all** is specified, all Passes are checked. If **[]** is specified, no Pass is checked and only tensor_graph is checked. If an invalid name is specified, it is ignored. <br> Type: List[str] <br> Default Value: empty |
| **pass_verify_error_tol**           | Input      | Meaning: **rtol** and **atol** used by the precision tool for precision comparison. <br> Description: The first value in the list is **rtol** and the second is **atol**. If the list length is not **2**, the default value is used. <br> Type: List[float] <br> Default Value: [1e-3, 1e-3] |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

## Example

```python
verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "/LARGE/DRIVE/DIR",
        }
pypto.set_verify_options(**verify_options)
```
