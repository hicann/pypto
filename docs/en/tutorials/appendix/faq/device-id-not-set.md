# Device ID Not Set for Operator Execution

<!-- md-trans-meta sourceCommit=1cc26711dfa4dcb46fee694d2efa62bd31a4f6ad translatedAt=2026-08-11T08:58:45.157Z pushedAt=2026-08-26T10:40:08.748Z -->

## Symptom

When an operator is executed on the Ascend AI processor, the execution fails and the following error message is displayed:

```text
2025-12-17 14:31:32.491 E | fail get device id, check if set device id
2025-12-17 14:31:32.492 E | RuntimeAgent::AllocDevAddr failed for size 20448
2025-12-17 14:31:32.493 E | RuntimeAgent::AllocDevAddr failed for size 20448
2025-12-17 14:31:32.493 E | aclmdlRICaptureGetInfo failed, return[100000]
```

## Possible Causes

The user-defined operator is not decorated with `@jit`, and the device ID for the current operator execution is not explicitly set using the TorchNPU API.

## Solution

Set the device ID before executing the operator. For example:

```python
def test_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))  # Obtain the expected device ID from the environment variable.
    torch.npu.set_device(device_id)  # Explicitly set the device ID.
    ....
```
