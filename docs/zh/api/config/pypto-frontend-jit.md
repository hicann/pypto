# pypto.frontend.jit

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`pypto.frontend.jit` 是前端架构中的核心装饰器，用于将 Python 函数即时编译（JIT）为高效的计算图并在 NPU 上执行。前端不支持返回值，仅支持 in-place 修改；支持传入 torch 张量及其他类型的变量。

主要特性：

- **In-place 修改**: 内核函数通过 in-place 修改输出张量传递计算结果，不支持返回值
- **类型注解**: 在函数签名中明确指定张量的形状和数据类型
- **直接调用**: 测试时可直接传入 torch 张量及其他类型的变量，无需显式转换
- **动态形状支持**: 配合 `pypto.DYNAMIC` 支持运行时变化的维度
- **多运行模式**: 支持 NPU 和 SIM（模拟器）两种运行模式

## 函数原型

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

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|----------|------|
| func | 输入 | frontend.jit 修饰的函数，kernel 入口，描述计算过程，用于构建计算图。 |
| host_options | 输入 | 类型为 `dict[str, any]`，用于设置 host 配置项，配置项参数见[参数说明](./pypto-set_host_options.md) |
| runtime_options | 输入 | 类型为 `dict[str, any]`，用于设置 runtime 配置项，配置项参数见[runtime_options 参数说明](#runtime_options_detail) |
| codegen_options | 输入 | 类型为 `dict[str, any]`，用于设置 codegen 配置项，配置项参数见[参数说明](./pypto-set_codegen_options.md)  |
| pass_options | 输入 | 类型为 `dict[str, any]`，用于设置 Pass 配置项，配置项参数见[参数说明](./pypto-set_pass_options.md)  |
| verify_options | 输入 | 类型为 `dict[str, any]`，用于设置 Verify 配置项，配置项参数见[参数说明](./pypto-set_verify_options.md) |
| debug_options | 输入 | 类型为 `dict[str, any]`，用于设置 debug 配置项，配置项参数见[参数说明](./pypto-set_debug_options.md) |

### runtime_options 参数说明 <a id="runtime_options_detail"></a>

| 参数名                         | 说明                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| device_sched_mode               | 含义：设置计算子图的调度模式 <br> 说明：0：代表默认调度模式，ready子图放入共享队列，各个调度线程抢占子图进行发送，子图获取发送遵循先入先出； <br> 1：代表L2cache亲和调度模式，选择最新依赖ready的子图优先下发，达到复用L2cache的效果； <br> 2：公平调度模式，aicpu上多线程调度管理多个aicore的时候，下发子图会尽量控制在多线程间的公平性，此模式会带来额外的调度管理开销； <br> 3：代表同时开启L2cache亲和调度模式以及公平调度模式； <br> 类型：int <br> 取值范围：0 或 1 或 2 或 3 <br> 默认值：0 <br> 影响pass范围：NA |
| stitch_function_max_num        | 含义：machine运行时ctrlflow aicpu里控制每次提交给schedule aicpu处理的最大device task的计算任务量 <br> 说明：设置的值代表每一个stitch task里处理的最大loop个数，该数值越大，通常stitch batch内并行度越高，相应的workspace内存使用也越大。<br> 类型：int <br> 取值范围:1 ~ 1024 <br> 默认值：128 <br> 影响pass范围：NA |
| run_mode                       | 含义：设置计算子图的执行设备 <br> 说明：<br> 0：表示在NPU上执行 <br> 1：表示在模拟器上执行 <br> 类型：int <br> 取值范围：0或者1 <br> 默认值：根据是否设置cann的环境变量来决定。如果设置了环境变量，则在NPU上执行；否则在模拟器上执行 <br> 影响pass范围：NA |
| valid_shape_optimize            | 含义：动态shape场景，validshape编译优化选项，打开该选项后，动态轴的Loop循环中，主块（shape与validshape相等）采用静态shape编译，尾块采用动态shape编译 <br> 说明：<br> 0：默认值，表示关闭validshape编译优化选项，所有Loop循环均采用动态shape进行编译 <br> 1：表示打开validshape编译优化选项 <br> 类型：int <br> 取值范围：0或者1 <br> 默认值：0 <br> 影响pass范围：NA |
| ready_on_host_tensors           | 含义：标记在Host端准备好的Kernel入口函数的输入tensor名称列表，格式为["tensor1", "tensor2", ...]。<br> 说明：如果算子的计算逻辑对某输入tensor有值依赖(即获取了tensor的值)，且此tensor的device数据在Host端已提前准备好，那么cpu的控制流可以提前发射以提升性能。<br> 类型：list of string <br> 默认值：空列表 <br> 影响pass范围：NA |
| device_sched_parallelism        | 含义：当算子中 pypto.loop 设置了可并行标记(parallel=True)时,此配置项用于指定 pypto.loop 在调度执行时的并行度 <br> 说明：使用此配置项前，请确保标记为可并行的 pypto.loop 的各个迭代之间不存在任何依赖关系，满足并行调度的条件。当并行度大于 1 时，该 pypto.loop 的多个迭代任务将被并发调度执行。需要注意的是，并行度数值越大，所需的 workspace 内存使用量也越大，通常与设置的并行度成倍数关系。<br> 类型：int <br> 取值范围:1 ~ 8 <br> 默认值： 1 <br> 影响pass范围：NA |
| launch_sched_aicpu_num        | 含义：指定启动的Schedule AICPU 线程数量 <br> 说明：当指定的数量大于硬件最大可用aicpu数量或者小于等于0时, 将启用硬件自动计算值，当前硬件类型为DAV2201时，最大可用aicpu数量为5，硬件类型为DAV3510时，最大可用aicpu数量为6（多款DAV3510芯片的最大值，具体最大数量取决于具体的型号）。<br> 类型：int <br> 取值范围:1 ~ 6 <br> 默认值： 6 <br> 影响pass范围：NA |
| launch_sched_same_cluster        | 含义：是否强制同Cluster线程分配 <br> 说明：0：允许跨Cluster，此时不需要多launch aicpu，可以减少aicpu使用开销，避免可能因为aicpu资源不够导致功能问题，但是性能会受影响，整体性能不及同Cluster线程分配场景； <br> 1：强制同Cluster，此时需要更多的aicpu资源来保证线程同Cluster分配，可能会导致aicpu资源不够，但是此模式下会提升调度性能，此外，当开启同cluster时, launch_sched_aicpu_num配置不生效； <br> 类型：int <br> 取值范围：0 或 1 <br> 默认值：1 <br> 影响pass范围：NA |
| launch_early_mode        | 含义：aicpu提前发射模式，支持aicpu不等待aicore启动后再启动 <br> 说明：当开启提前发射后，可以减少aicpu启动头开销，提升性能，但是aicpu提前发射会提前占用aicpu资源，在接入整网或者hccl用aicpu做通信域展开时会存在aicpu由于竞争而资源不够的情况，可能会导致功能问题。0：仅capture模式提前发射； <br> 1：所有模式都提前发射； <br> 2：所有模式都不提前发射 <br> 类型：int <br> 取值范围:0 ~ 2 <br> 默认值： 0 <br> 影响pass范围：NA |

## 返回值说明

返回装饰后的函数，该函数可被直接调用执行。

## 约束说明

1. 张量参数，必须使用类型注解指定为 `pypto.Tensor` 类型
2. 动态维度必须使用 `pypto.DYNAMIC` 或 `pypto.DYN` 在参数注解中标记，未标记时，默认按静态维度处理
3. tensor format用format标记，format支持非显式标记(参考示例1中的a), 默认为pypto.TileOpFormat.TILEOP_ND;
   format显式标记时, 性能更优, 要求传入的torch tensor与pypto.Tensor声明的format一致，能获得更优的性能;
4. 张量参数在前，非张量参数（如 `scalar`、`tiling`）在后
5. 非张量参数支持 keyword 传参、位置参数、使用默认值

**pypto.Tensor[...]说明**：

- kernel函数里申明推荐使用 `pypto.Tensor[[shape], dtype]` 方括号语法，符合 Python 类型注解规范
- 也兼容旧的小括号语法 `pypto.Tensor([shape], dtype)`
- 方括号内不支持 `key=value` 形式的关键字参数（Python 语法限制），只能按位置传递或使用字典
- `pypto.Tensor[]`（空参数）不支持

## 调用示例

### 示例1: 基础使用

```python
@pypto.frontend.jit
def add_kernel(
    a: pypto.Tensor([3], pypto.DT_FP32),
    b: pypto.Tensor([3], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_NZ),
    out: pypto.Tensor([3], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


# 直接传入 torch 张量调用
x = torch.randn(3, dtype=torch.float32, device='npu:0')
y = torch.randn(3, dtype=torch.float32, device='npu:0')
result = add_kernel(x, y)
```

### 示例2: 指定运行模式

```python
# NPU 模式
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel_npu(x: pypto.Tensor):
    ...

# Cost Model 模式
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def kernel_sim(x: pypto.Tensor):
    ...
```
