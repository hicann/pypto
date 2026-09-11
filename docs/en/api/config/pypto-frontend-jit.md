# pypto.frontend.jit

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:07:00.316Z pushedAt=2026-08-26T09:10:38.127Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

`pypto.frontend.jit` is the core decorator in the frontend architecture. It just-in-time (JIT) compiles Python functions into efficient computation graphs and executes them on the NPU. The frontend does not support return values and only supports in-place modification. It supports passing in torch tensors and variables of other types.

Key features:

- **In-place modification**: Kernel functions pass computation results by modifying output tensors in place, and return values are not supported.
- **Type annotation**: Explicitly specify the shape and data type of tensors in the function signature.
- **Direct call**: During testing, torch tensors and other types of variables can be passed directly without explicit conversion.
- **Dynamic shape support**: Works with `pypto.DYNAMIC` to support dimensions that vary at runtime.
- **Multiple run modes**: Supports both NPU and SIM (simulator) run modes.

## Prototype

```python
@pypto.frontend.jit(
    host_options=None,
    runtime_options=None,
    codegen_options=None,
    pass_options=None
)
def kernel_function(...):
    ...
```

## Parameters

| Parameter | Input/Output | Description |
|--------|----------|------|
| func | Input | Function decorated by **frontend.jit**, serving as the kernel entry, which describes the computation process and is used to build the computation graph. |
| host_options | Input | The type is `dict[str, any]`. Used to set host configuration items. For configuration item parameters, see [Parameters](./pypto-set_host_options.md). |
| runtime_options | Input | The type is `dict[str, any]`. Used to set runtime configuration items. For configuration item parameters, see [runtime_options Parameters](#runtime_options_detail). |
| codegen_options | Input | The type is `dict[str, any]`. Used to set codegen configuration items. For configuration item parameters, see [Parameters](./pypto-set_codegen_options.md). |
| pass_options | Input | The type is `dict[str, any]`. Used to set **pass** configuration items. For configuration item parameters, see [Parameters](./pypto-set_pass_options.md). |
| verify_options | Input | The type is `dict[str, any]`. Used to set **verify** configuration items. For configuration item parameters, see [Parameters](./pypto-set_verify_options.md). |
| debug_options | Input | The type is `dict[str, any]`. Used to set **debug** configuration items. For configuration item parameters, see [Parameters](./pypto-set_debug_options.md). |

### runtime_options Parameters<a id="runtime_options_detail"></a>

| Parameter                         | Description                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| **device_sched_mode**               | Meaning: Sets the scheduling mode of the computation subgraph. <br> Description: <br>**0**: Default scheduling mode. Ready subgraphs are placed into a shared queue, and each scheduling thread preempts subgraphs for dispatch. Subgraph acquisition and dispatch follow first-in-first-out order. <br> **1**: L2cache affinity scheduling mode. The subgraph whose latest dependency becomes ready is dispatched first to reuse the L2cache. <br> **2**: Fair scheduling mode. When multiple threads on the AICPU schedule and manage multiple AI Cores, subgraph dispatch is controlled as fairly as possible across threads. This mode introduces additional scheduling management overhead. <br> **3**: Enables both the L2cache affinity scheduling mode and the fair scheduling mode. <br> Type: int <br> Value Range: 0, 1, 2, or 3 <br> Default Value: 0 <br> Affected pass scope: N/A |
| **stitch_function_max_num**        | Meaning: Controls the maximum computation workload of device tasks submitted to the schedule AICPU for processing at a time in the ctrlflow AICPU during machine runtime. <br> Description: The configured value represents the maximum number of loops processed in each stitch task. A larger value generally means higher parallelism within a stitch batch, and correspondingly greater workspace memory usage. <br> Type: int <br> Value Range: 1 to 1024 <br> Default Value: 128 <br> Affected pass scope: N/A |
| **run_mode**                       | Meaning: Sets the execution device of the computation subgraph. <br> Description: <br>**0**: Executes on the NPU. <br> **1**: Executes on the simulator. <br> Type: int <br> Value Range: 0 or 1 <br> Default Value: Determined by whether the CANN environment variable is set. If the environment variable is set, execution runs on the NPU; otherwise, execution runs on the simulator. <br> Affected pass scope: N/A |
| **valid_shape_optimize**            | Meaning: validshape compilation optimization option for dynamic shape scenarios. When this option is enabled, in the loop of a dynamic axis, the main block (where shape equals validshape) is compiled with static shape, and the tail block is compiled with dynamic shape. <br> Description: <br>**0**: Default value, which disables the validshape compilation optimization option. All Loops are compiled with dynamic shape. <br> **1**: Enables the validshape compilation optimization option. <br> Type: int <br> Value Range: 0 or 1 <br> Default Value: 0 <br> Affected pass scope: N/A |
| **ready_on_host_tensors**           | Meaning: Marks the list of input tensor names of the Kernel entry function that are ready on the host side, in the format ["tensor1", "tensor2", ...]. <br> Description: If the operator's computation logic has a value dependency on an input tensor (that is, it obtains the tensor's value), and the device data of this tensor is already prepared on the host side, the CPU control flow can be launched early to improve performance. <br> Type: list of string <br> Default Value: empty list <br> Affected pass scope: N/A |
| **device_sched_parallelism**        | Meaning: When **pypto.loop** in the operator is marked as parallelizable (**parallel=True**), this configuration item specifies the parallelism of **pypto.loop** during scheduling and execution. <br> Description: Before using this configuration item, ensure that there are no dependencies between the iterations of the **pypto.loop** marked as parallelizable, so that the conditions for parallel scheduling are met. When the parallelism is greater than 1, multiple iteration tasks of this **pypto.loop** are scheduled and executed concurrently. Note that a larger parallelism value requires more workspace memory, which is generally proportional to the configured parallelism. <br> Type: int <br> Value Range: 1 to 8 <br> Default Value: 1 <br> Affected pass scope: N/A |
| **launch_sched_aicpu_num**        | Meaning: Specifies the number of Schedule AICPU threads to launch. <br> Description: When the specified number is greater than the maximum available AICPU count of the hardware or less than or equal to 0, the hardware automatically calculated value is used. For Atlas A3 training products/Atlas A3 inference products/Atlas A2 training products/Atlas A2 inference products, the maximum available AICPU count is 5. For Ascend 950PR/Ascend 950DT, the maximum available AICPU count is 7 (the specific maximum count depends on the specific model). <br> Type: int <br> Value Range: 1 to 7 <br> Default Value: 7 <br> Affected pass scope: N/A |
| **launch_early_mode**        | Meaning: AICPU early launch mode, which allows the AICPU to start without waiting for the AICore to start first. <br> Description: When early launch is enabled, the AICPU startup overhead is reduced and performance is improved. However, early AICPU launch occupies AICPU resources in advance. When integrating into the entire network or when HCCL uses the AICPU for communicator expansion, the AICPU may run out of resources due to contention, which may cause functional issues. <br>**0**: Early launch only in capture mode. <br> **1**: Early launch in all modes. <br> **2**: No early launch in any mode. <br> Type: int <br> Value Range: 0 to 2 <br> Default Value: 0 <br> Affected pass scope: N/A |

## Return Value

Returns the decorated function, which can be called directly for execution.

## Constraints

1. Tensor parameters must be annotated with the `pypto.Tensor` type using type annotations.
2. Dynamic dimensions must be marked with `pypto.DYNAMIC` or `pypto.DYN` in the parameter annotations. If not marked, they are processed as static dimensions by default.
3. The tensor format is marked with **format**. **format** supports non-explicit marking (see `a` in Example 1), which defaults to `pypto.TileOpFormat.TILEOP_ND`.
   When **format** is explicitly marked, better performance can be achieved, and the passed torch tensor must be consistent with the format declared by `pypto.Tensor` to obtain better performance.
4. Tensor parameters come first, followed by non-tensor parameters (such as `scalar` and `tiling`).
5. Non-tensor parameters support keyword arguments, positional arguments, and default values.

**Description of `pypto.Tensor[...]`**:

- In kernel functions, it is recommended to declare tensors using the `pypto.Tensor[[shape], dtype]` bracket syntax, which conforms to the Python type annotation specification.
- The legacy parenthesis syntax `pypto.Tensor([shape], dtype)` is also supported.
- Keyword arguments in the `key=value` form are not supported inside square brackets (due to Python syntax restrictions); they can only be passed by position or using a dictionary.
- `pypto.Tensor[]` (with empty arguments) is not supported.

## Example

### Example 1: Basic Usage

```python
@pypto.frontend.jit
def add_kernel(
    a: pypto.Tensor([3], pypto.DT_FP32),
    b: pypto.Tensor([3], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_NZ),
    out: pypto.Tensor([3], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


# Directly pass a torch tensor for calling.
x = torch.randn(3, dtype=torch.float32, device='npu:0')
y = torch.randn(3, dtype=torch.float32, device='npu:0')
result = add_kernel(x, y)
```

### Example 2: Specifying the Run Mode

```python
# NPU mode.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel_npu(x: pypto.Tensor):
    ...

# Cost Model mode.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def kernel_sim(x: pypto.Tensor):
    ...
```
